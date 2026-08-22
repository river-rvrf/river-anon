//! Canonical byte encodings — port of `river-py/codec.py`.
//!
//! Two jobs: serialize public keys, proofs and public parameters so they
//! compare byte-for-byte against the Python reference, and provide the
//! canonical transcript fed to the Fiat–Shamir hash `H`.
//!
//! Everything goes through one bit-level codec.  A [`Layout`] is an
//! ordered list of [`Field`]s; each field names a [`Coder`], and the same
//! list drives both directions, so an encoder and a decoder cannot drift
//! apart.  Three coders cover every quantity in the scheme:
//!
//! * [`Coder::Uniform`] — values in `[0, modulus)` at exactly
//!   `bit_length(modulus - 1)` bits: commitments, high bits, ring elements.
//! * [`Coder::Signed`] — centred values with `|v| <= bound`,
//!   offset-encoded at `bit_length(2·bound)` bits: challenges, ternary
//!   randomness, anything uniform on its range.
//! * [`Coder::Rice`] — centred discrete-Gaussian values, Golomb–Rice
//!   coded: every masked response, in both the OOM and exact layers.
//!
//! Rice is what the paper's size estimates assume: it charges
//! `h(sigma) = log2(4.13·sigma)` bits per Gaussian coefficient, and Rice
//! lands within about half a bit of that, against the 32 bits a
//! fixed-width field would spend on `z`.
//!
//! **Proof size is therefore data-dependent.**  A Gaussian coefficient far
//! from zero costs more bits than one near it, so two honest proofs at the
//! same profile differ in length.  [`Layout::max_bytes`] is the worst case
//! — every coefficient at its bound — and is what a fixed buffer must be
//! sized to.
//!
//! ## Two things this port cannot choose freely
//!
//! Bit order is least-significant-first within each byte, and the whole
//! layout is padded to a byte boundary *once*, at the end — not after each
//! polynomial, which is what the sibling `lotrs-rs` codec does.  Both are
//! forced by `vectors.json`.
//!
//! [`optimal_rice_k`] is evaluated in integers over the *exact rational*
//! sigma, never in floating point.  `k` is wire-visible: a half-ulp
//! disagreement between two implementations is a different encoding, not a
//! rounding difference.
//!
//! ## Robustness
//!
//! Decoding is hostile-input safe and total.  Every read checks
//! truncation, a value outside its declared range, a unary run longer than
//! the bound permits, a non-canonical residue `>= modulus`, nonzero
//! padding bits, and trailing bytes.  All of them return a [`CodecError`],
//! so a caller — `verify()`, above all — can turn any malformed proof into
//! a plain `false`.
//!
//! The claim is worth stating exactly, because "no decoder panics" was
//! too broad once and had to be narrowed twice:
//!
//! * **Anything taking bytes or values is total**, for *any* argument —
//!   including a nonsensical configuration.  A byte width past `u64`, a
//!   zero ring modulus, a zero-column field, a negative bound, a
//!   [`Coder`] variant built by struct literal that the constructors
//!   would have refused: each is [`CodecError::Unrepresentable`], not an
//!   abort.  This holds for arbitrary layouts, not only the profile-
//!   derived ones.
//! * **Constructors validate and panic.**  [`Coder::uniform`],
//!   [`Coder::signed`], [`Coder::rice_with_k`] and [`Layout::new`] take
//!   profile constants — never anything off the wire — and a nonsensical
//!   one is a programming error, documented per constructor.  The
//!   corresponding *use* sites still check, because the enum's fields are
//!   public and a caller can bypass the constructor.

use crate::fixed::Nat;
use crate::params::RiVeRParams;
use crate::ring::{Poly, Ring};
use crate::sample::{hash_bytes, rational_sigma, Part, DS_CHALLENGE};

/// Codec-level failure.  Carries the class of malformation and nothing
/// else: a caller holding attacker-supplied bytes should surface one bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecError {
    /// Ran out of bits or bytes mid-value.
    Truncated,
    /// Bytes left over after a complete, well-formed decode.
    TrailingBytes,
    /// Padding to the final byte boundary was not all zero.
    NonZeroPadding,
    /// A Rice unary run longer than the field's bound can produce.
    UnaryOverflow,
    /// A value outside the range its coder declares.
    OutOfRange,
    /// A residue `>= modulus`: representable in the field's width, but not
    /// the canonical representative, so it would re-encode differently.
    NonCanonical,
    /// A container with the wrong number of rows, columns or bytes.
    LengthMismatch,
    /// A length prefix no encoding at this profile could have produced.
    BadLengthPrefix,
    /// A field configuration this codec cannot represent: a byte width
    /// past `u64`, or a value too large for the width it was given.
    /// A caller error rather than a malformed input, but reported the
    /// same way so no entry point has to panic to say it.
    Unrepresentable,
}

impl core::fmt::Display for CodecError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            CodecError::Truncated => "truncated: ran out of bits",
            CodecError::TrailingBytes => "trailing bytes",
            CodecError::NonZeroPadding => "nonzero padding bits",
            CodecError::UnaryOverflow => "unary run exceeds the field bound",
            CodecError::OutOfRange => "value outside its declared range",
            CodecError::NonCanonical => "non-canonical residue",
            CodecError::LengthMismatch => "length mismatch",
            CodecError::BadLengthPrefix => "length prefix outside the profile's range",
            CodecError::Unrepresentable => "field configuration is not representable",
        })
    }
}

impl std::error::Error for CodecError {}

pub type Result<T> = core::result::Result<T, CodecError>;

// ---- bit-level I/O -------------------------------------------------------

/// Accumulates bits, least-significant-first within each byte.
#[derive(Clone, Debug, Default)]
pub struct BitWriter {
    out: Vec<u8>,
    /// Bits not yet flushed, LSB-aligned.  Always fewer than eight after
    /// any complete call, which is what keeps the shifts below in range.
    pend: u64,
    nbits: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            out: Vec::with_capacity(bytes),
            pend: 0,
            nbits: 0,
        }
    }

    pub fn write_bits(&mut self, value: u64, n: u32) {
        if n == 0 {
            return;
        }
        // `pend` holds at most seven bits, so a single shift is in range
        // only up to 56.  Wider fields — `Uniform` over the largest `q` is
        // 54 bits — stay under it; the split is here so nothing above can
        // silently truncate.
        if n > 56 {
            self.write_bits(value & 0xFFFF_FFFF, 32);
            self.write_bits(value >> 32, n - 32);
            return;
        }
        let masked = value & ((1u64 << n) - 1);
        self.pend |= masked << self.nbits;
        self.nbits += n;
        while self.nbits >= 8 {
            self.out.push((self.pend & 0xFF) as u8);
            self.pend >>= 8;
            self.nbits -= 8;
        }
    }

    /// `value` one-bits then a zero-bit, written in chunks so a long run
    /// does not cost one call per bit.
    pub fn write_unary(&mut self, mut value: u64) {
        while value >= 32 {
            self.write_bits(0xFFFF_FFFF, 32);
            value -= 32;
        }
        self.write_bits((1u64 << value) - 1, value as u32);
        self.write_bits(0, 1);
    }

    /// Flush, zero-padding to the next byte boundary.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = self.out.clone();
        if self.nbits != 0 {
            out.push((self.pend & 0xFF) as u8);
        }
        out
    }

    pub fn into_bytes(mut self) -> Vec<u8> {
        if self.nbits != 0 {
            self.out.push((self.pend & 0xFF) as u8);
            self.pend = 0;
            self.nbits = 0;
        }
        self.out
    }

    pub fn bit_length(&self) -> usize {
        8 * self.out.len() + self.nbits as usize
    }
}

/// Reads bits least-significant-first, rejecting every malformed input.
#[derive(Clone, Debug)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Next source byte to buffer.
    byte: usize,
    pend: u64,
    nbits: u32,
    /// Bits consumed by the caller, which is not `8 * byte`: up to seven
    /// buffered bits may be unread.
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte: 0,
            pend: 0,
            nbits: 0,
            pos: 0,
        }
    }

    pub fn read_bits(&mut self, n: u32) -> Result<u64> {
        if n == 0 {
            return Ok(0);
        }
        if n > 56 {
            let lo = self.read_bits(32)?;
            let hi = self.read_bits(n - 32)?;
            return Ok(lo | (hi << 32));
        }
        while self.nbits < n {
            if self.byte >= self.data.len() {
                return Err(CodecError::Truncated);
            }
            self.pend |= (self.data[self.byte] as u64) << self.nbits;
            self.byte += 1;
            self.nbits += 8;
        }
        let value = self.pend & ((1u64 << n) - 1);
        self.pend >>= n;
        self.nbits -= n;
        self.pos += n as usize;
        Ok(value)
    }

    /// Read a unary run, rejecting one longer than `max_value`.
    ///
    /// Without the cap, a crafted input of all-ones bytes makes the
    /// decoder spin until it exhausts the buffer; the cap turns that into
    /// an immediate error.
    pub fn read_unary(&mut self, max_value: u64) -> Result<u64> {
        let mut count = 0u64;
        while self.read_bits(1)? != 0 {
            count += 1;
            if count > max_value {
                return Err(CodecError::UnaryOverflow);
            }
        }
        Ok(count)
    }

    /// Require zero padding to the byte boundary and no trailing bytes.
    pub fn finish(&mut self) -> Result<()> {
        while self.pos & 7 != 0 {
            if self.read_bits(1)? != 0 {
                return Err(CodecError::NonZeroPadding);
            }
        }
        if (self.pos >> 3) != self.data.len() {
            return Err(CodecError::TrailingBytes);
        }
        Ok(())
    }

    pub fn bit_pos(&self) -> usize {
        self.pos
    }
}

// ---- coders --------------------------------------------------------------

/// Rice parameter for a discrete Gaussian of width `num / den`.
///
/// The classical choice `k = floor(log2(sqrt(2 ln 2)·sigma))`, evaluated
/// in integers over the exact rational the sampler uses, so no float
/// rounding can make two implementations pick different `k` and produce
/// incompatible bytes.
pub fn optimal_rice_k(sigma_num: u64, sigma_den: u64) -> u32 {
    // `sqrt(2 ln 2)` to 30 significant figures.  This used to be
    // `11774/10000`, a relative error of `8.5e-6` — fine in practice and
    // unchecked in principle, since `k` moves whenever the true and the
    // approximate products straddle a power of two.  Thirty digits makes
    // that window `1e-30` wide, and
    // `manifest::tests::rice_parameters_are_far_from_a_boundary`
    // measures the actual distance at every field of every profile.
    //
    // The product leaves `u128` — `1.18e30 · 2.9e12` is `3.4e42` — so it
    // goes through [`Nat`], which is also what the reference's
    // arbitrary-precision integers do.  Only setup calls this; the wire
    // path reads `k` from [`crate::manifest`].
    let c = Nat::from_dec_str(crate::manifest::RICE_CONST_NUM_DEC)
        .expect("RICE_CONST_NUM_DEC is decimal");
    let den = Nat::from_dec_str(crate::manifest::RICE_CONST_DEN_DEC)
        .expect("RICE_CONST_DEN_DEC is decimal")
        .mul_u64(sigma_den);
    let scaled = c.mul_u64(sigma_num).div(&den);
    scaled.bit_len().saturating_sub(1)
}

/// [`optimal_rice_k`] on a float width, pinned through [`rational_sigma`]
/// first — the same pinning the sampler applies.
pub fn optimal_rice_k_f(sigma: f64) -> u32 {
    let (num, den) = rational_sigma(sigma);
    optimal_rice_k(num, den)
}

/// How one integer becomes bits.
///
/// Constructors take parameters that come from a profile, never from the
/// wire, and panic on a nonsensical one; every *decode* path is total.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coder {
    /// Values in `[0, modulus)` at exactly `width` bits.
    Uniform { modulus: u64, width: u32 },
    /// Centred values with `|v| <= bound`, offset-encoded at `width` bits.
    Signed { bound: i64, width: u32 },
    /// Golomb–Rice: `k` low bits, the high part in unary, then a sign bit
    /// that is *omitted when the value is zero*.
    ///
    /// Omitting the sign of zero is what keeps the encoding canonical:
    /// `-0` and `+0` are the same integer and must have the same image, or
    /// decode-after-encode would not be the identity on bytes.
    Rice { k: u32, bound: i64, max_high: u64 },
}

impl Coder {
    /// # Panics
    ///
    /// On `modulus == 0`, and on a modulus past `i64::MAX`.  The upper
    /// limit is real, not defensive: [`Coder::read`] returns `i64`, so a
    /// modulus above `2^63` has representable residues that come back
    /// negative and cannot be re-encoded — `u64::MAX - 1` would decode as
    /// `-2`.  Every RiVeR modulus is under `2^55`, and a coder is built
    /// from profile constants, never from the wire, so this is a
    /// constructor precondition rather than a runtime error.
    pub fn uniform(modulus: u64) -> Self {
        assert!(modulus >= 1, "modulus must be positive");
        assert!(
            modulus <= i64::MAX as u64,
            "modulus {modulus} exceeds the i64 the coder decodes into"
        );
        let width = (64 - (modulus - 1).leading_zeros()).max(1);
        Coder::Uniform { modulus, width }
    }

    pub fn signed(bound: i64) -> Self {
        assert!(bound >= 0, "bound must be non-negative");
        let width = (64 - (2 * bound as u64).leading_zeros()).max(1);
        Coder::Signed { bound, width }
    }

    /// Rice with an explicit parameter, for a layout that pins `k` rather
    /// than deriving it.
    pub fn rice_with_k(k: u32, bound: i64) -> Self {
        assert!(bound >= 0, "bound must be non-negative");
        assert!(k < 63, "Rice parameter out of range");
        Coder::Rice {
            k,
            bound,
            max_high: (bound >> k) as u64 + 1,
        }
    }

    /// Rice at the parameter [`optimal_rice_k`] picks for the exact
    /// rational `num / den`.
    pub fn rice(sigma_num: u64, sigma_den: u64, bound: i64) -> Self {
        Coder::rice_with_k(optimal_rice_k(sigma_num, sigma_den), bound)
    }

    /// Rice at a float width, pinned to a rational first.
    pub fn rice_sigma(sigma: f64, bound: i64) -> Self {
        let (num, den) = rational_sigma(sigma);
        Coder::rice(num, den, bound)
    }

    pub fn write(&self, w: &mut BitWriter, value: i64) -> Result<()> {
        match *self {
            Coder::Uniform { modulus, width } => {
                if value < 0 || value as u64 >= modulus {
                    return Err(CodecError::OutOfRange);
                }
                w.write_bits(value as u64, width);
            }
            Coder::Signed { bound, width } => {
                if value < -bound || value > bound {
                    return Err(CodecError::OutOfRange);
                }
                // Through `i128`: `value + bound` is up to `2·bound`, which
                // overflows `i64` for a bound past `2^62`.  No profile
                // reaches one, but a coder is constructible with one and
                // this path must not depend on that.
                w.write_bits((value as i128 + bound as i128) as u64, width);
            }
            Coder::Rice { k, bound, .. } => {
                // The variants are public data, so a caller can build a
                // `Rice` the constructors would have refused; `1 << k`
                // would then shift past the word.
                if k >= 63 || bound < 0 {
                    return Err(CodecError::Unrepresentable);
                }
                let magnitude = value.unsigned_abs();
                if magnitude > bound as u64 {
                    return Err(CodecError::OutOfRange);
                }
                w.write_bits(magnitude & ((1u64 << k) - 1), k);
                w.write_unary(magnitude >> k);
                if magnitude != 0 {
                    w.write_bits(u64::from(value < 0), 1);
                }
            }
        }
        Ok(())
    }

    pub fn read(&self, r: &mut BitReader<'_>) -> Result<i64> {
        match *self {
            Coder::Uniform { modulus, width } => {
                let value = r.read_bits(width)?;
                if value >= modulus {
                    return Err(CodecError::NonCanonical);
                }
                Ok(value as i64)
            }
            Coder::Signed { bound, width } => {
                // A 64-bit field read back as `i64` can be negative before
                // the offset is removed, so the subtraction happens in
                // `i128` — a decoder may not overflow on hostile bytes.
                let value = r.read_bits(width)? as i128 - bound as i128;
                if value < -(bound as i128) || value > bound as i128 {
                    return Err(CodecError::OutOfRange);
                }
                Ok(value as i64)
            }
            Coder::Rice { k, bound, max_high } => {
                if k >= 63 || bound < 0 {
                    return Err(CodecError::Unrepresentable);
                }
                let low = r.read_bits(k)?;
                let high = r.read_unary(max_high)?;
                let magnitude = (high << k) | low;
                if magnitude > bound as u64 {
                    return Err(CodecError::OutOfRange);
                }
                if magnitude == 0 {
                    return Ok(0);
                }
                let negative = r.read_bits(1)? != 0;
                Ok(if negative {
                    -(magnitude as i64)
                } else {
                    magnitude as i64
                })
            }
        }
    }

    /// Worst case: a fixed field is its width; a Rice value sits exactly
    /// at the bound, paying `k` low bits, then the unary of `bound >> k`
    /// — one bit per unit plus the terminating zero — then a sign bit.
    ///
    /// Not `max_high`, which is deliberately one larger than any
    /// encodable high part: it is the cap [`BitReader::read_unary`]
    /// enforces, and a permissive cap is the right kind because the
    /// magnitude check behind it rejects the extra value. Charging that
    /// slack as a real bit overstated every layout by one bit per Rice
    /// coefficient — 620 bytes at `RiVeR-N8`, 1648 at `RiVeR-N256`. Only
    /// size metadata and the framing bound move; no encoded byte does.
    pub fn max_bits(&self) -> usize {
        match *self {
            Coder::Uniform { width, .. } | Coder::Signed { width, .. } => width as usize,
            // `+ 1` for the unary terminator, and the sign bit only if
            // some encodable value is nonzero — at `bound == 0` the sole
            // value is `0`, whose sign is omitted to keep the encoding
            // canonical, so it costs `k + 1` and not `k + 2`.
            Coder::Rice { k, bound, .. } => {
                k as usize + (bound >> k) as usize + 1 + usize::from(bound != 0)
            }
        }
    }

    /// Best case, reached when a Rice value is zero.
    pub fn min_bits(&self) -> usize {
        match *self {
            Coder::Uniform { width, .. } | Coder::Signed { width, .. } => width as usize,
            Coder::Rice { k, .. } => k as usize + 1,
        }
    }
}

// ---- layouts -------------------------------------------------------------

/// One field's value.
///
/// The variant has to match the field: a field with a ring carries
/// canonical residues and is centred in transit, everything else carries
/// integers the coder sees directly.  Mixing them is a
/// [`CodecError::LengthMismatch`], not a silent reinterpretation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FieldValue {
    /// Plain integers: one row for a flat field, `rows` rows otherwise.
    Ints(Vec<Vec<i64>>),
    /// Canonical residues in `[0, q)` of the field's ring.
    Residues(Vec<Poly>),
}

impl FieldValue {
    /// A flat field's single row of integers.
    pub fn flat(values: Vec<i64>) -> Self {
        FieldValue::Ints(vec![values])
    }

    pub fn as_ints(&self) -> Result<&[Vec<i64>]> {
        match self {
            FieldValue::Ints(v) => Ok(v),
            FieldValue::Residues(_) => Err(CodecError::LengthMismatch),
        }
    }

    pub fn as_residues(&self) -> Result<&[Poly]> {
        match self {
            FieldValue::Residues(v) => Ok(v),
            FieldValue::Ints(_) => Err(CodecError::LengthMismatch),
        }
    }

    /// The single row of a flat integer field.
    pub fn as_flat(&self) -> Result<&[i64]> {
        match self {
            FieldValue::Ints(v) if v.len() == 1 => Ok(&v[0]),
            _ => Err(CodecError::LengthMismatch),
        }
    }

    fn rows(&self) -> usize {
        match self {
            FieldValue::Ints(v) => v.len(),
            FieldValue::Residues(v) => v.len(),
        }
    }
}

/// One named entry of a [`Layout`].
///
/// `rows == None` means the value is a single row of `cols` integers;
/// otherwise it is `rows` rows of `cols`.  `ring_q`, when set, means the
/// value holds residues of `R_{ring_q}`: they are centred on the way out
/// and lifted back on the way in, so the coder only ever sees small signed
/// integers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    pub name: &'static str,
    pub coder: Coder,
    pub rows: Option<usize>,
    pub cols: usize,
    pub ring_q: Option<u64>,
}

impl Field {
    pub fn flat(name: &'static str, coder: Coder, cols: usize) -> Self {
        Self {
            name,
            coder,
            rows: None,
            cols,
            ring_q: None,
        }
    }

    pub fn rows(name: &'static str, coder: Coder, cols: usize, rows: usize) -> Self {
        Self {
            name,
            coder,
            rows: Some(rows),
            cols,
            ring_q: None,
        }
    }

    pub fn ring_rows(name: &'static str, coder: Coder, cols: usize, rows: usize, q: u64) -> Self {
        Self {
            name,
            coder,
            rows: Some(rows),
            cols,
            ring_q: Some(q),
        }
    }

    pub fn count(&self) -> usize {
        self.cols * self.rows.unwrap_or(1)
    }
}

/// An ordered field list that drives encoding and decoding alike.
///
/// Values are positional, in field order; [`Layout::index_of`] recovers a
/// field by name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub fields: Vec<Field>,
}

impl Layout {
    pub fn new(fields: Vec<Field>) -> Self {
        for (i, f) in fields.iter().enumerate() {
            assert!(
                !fields[..i].iter().any(|g| g.name == f.name),
                "duplicate field name in layout: {}",
                f.name
            );
        }
        Self { fields }
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }

    pub fn encode(&self, values: &[FieldValue]) -> Result<Vec<u8>> {
        if values.len() != self.fields.len() {
            return Err(CodecError::LengthMismatch);
        }
        let mut w = BitWriter::with_capacity(self.max_bytes());
        for (f, value) in self.fields.iter().zip(values) {
            if value.rows() != f.rows.unwrap_or(1) {
                return Err(CodecError::LengthMismatch);
            }
            match (f.ring_q, value) {
                (None, FieldValue::Ints(rows)) => {
                    for row in rows {
                        if row.len() != f.cols {
                            return Err(CodecError::LengthMismatch);
                        }
                        for &c in row {
                            f.coder.write(&mut w, c)?;
                        }
                    }
                }
                (Some(q), FieldValue::Residues(rows)) => {
                    // A zero modulus names no residue class.  Encoding
                    // happened to refuse it already — every `c >= 0` — but
                    // by accident, and decoding divided by zero.
                    if q == 0 {
                        return Err(CodecError::Unrepresentable);
                    }
                    let half = q / 2;
                    for row in rows {
                        if row.len() != f.cols {
                            return Err(CodecError::LengthMismatch);
                        }
                        for &c in row {
                            // Centring assumes its input is already in
                            // `[0, q)`: given `c + q` it would happily
                            // return `c`, so a non-canonical residue would
                            // encode as the canonical one and verify.
                            // Reject instead, matching the byte decoder.
                            if c >= q {
                                return Err(CodecError::NonCanonical);
                            }
                            let centred = if c > half {
                                c as i64 - q as i64
                            } else {
                                c as i64
                            };
                            f.coder.write(&mut w, centred)?;
                        }
                    }
                }
                _ => return Err(CodecError::LengthMismatch),
            }
        }
        Ok(w.into_bytes())
    }

    pub fn decode(&self, data: &[u8]) -> Result<Vec<FieldValue>> {
        let mut r = BitReader::new(data);
        let mut out = Vec::with_capacity(self.fields.len());
        for f in &self.fields {
            let n_rows = f.rows.unwrap_or(1);
            match f.ring_q {
                None => {
                    let mut rows = Vec::with_capacity(n_rows);
                    for _ in 0..n_rows {
                        let mut row = Vec::with_capacity(f.cols);
                        for _ in 0..f.cols {
                            row.push(f.coder.read(&mut r)?);
                        }
                        rows.push(row);
                    }
                    out.push(FieldValue::Ints(rows));
                }
                Some(q) => {
                    if q == 0 {
                        return Err(CodecError::Unrepresentable);
                    }
                    let mut rows = Vec::with_capacity(n_rows);
                    for _ in 0..n_rows {
                        let mut row = Vec::with_capacity(f.cols);
                        for _ in 0..f.cols {
                            row.push(f.coder.read(&mut r)?.rem_euclid(q as i64) as u64);
                        }
                        rows.push(row);
                    }
                    out.push(FieldValue::Residues(rows));
                }
            }
        }
        r.finish()?;
        Ok(out)
    }

    /// Upper bound on the encoding: every value at its worst case.
    pub fn max_bytes(&self) -> usize {
        let bits: usize = self
            .fields
            .iter()
            .map(|f| f.count() * f.coder.max_bits())
            .sum();
        bits.div_ceil(8)
    }

    /// Lower bound, reached when every Rice-coded value is zero.
    pub fn min_bytes(&self) -> usize {
        let bits: usize = self
            .fields
            .iter()
            .map(|f| f.count() * f.coder.min_bits())
            .sum();
        bits.div_ceil(8)
    }
}

// ---- primitive packing ---------------------------------------------------

/// Bytes needed for an unsigned value in `[0, modulus)`.
pub fn width_for_modulus(modulus: u64) -> usize {
    ((64 - modulus.leading_zeros()) as usize).div_ceil(8)
}

/// Bytes needed for a signed value in `[-bound, bound]`.
///
/// `Err` for a negative bound, which names no range: computing it anyway
/// overflowed `2 · bound as u64` and aborted in a debug build.
pub fn width_for_bound(bound: i64) -> Result<usize> {
    if bound < 0 {
        return Err(CodecError::Unrepresentable);
    }
    let span = 2 * bound as u128 + 1;
    Ok(((128 - span.leading_zeros()) as usize).div_ceil(8))
}

/// Pack unsigned values at `width` bytes each, little-endian.
///
/// A value too large for `width` is an error, not a truncation — the
/// reference raises `OverflowError` there, and silently dropping the high
/// bytes would produce a shorter-looking value that re-encodes to
/// different bytes.  A `width` past `u64` is
/// [`CodecError::Unrepresentable`] rather than a panic, so no entry point
/// in this module aborts on a bad configuration.
pub fn pack_unsigned(values: &[u64], width: usize) -> Result<Vec<u8>> {
    if width > 8 {
        return Err(CodecError::Unrepresentable);
    }
    let mut out = Vec::with_capacity(values.len() * width);
    for &v in values {
        let bytes = v.to_le_bytes();
        if bytes[width..].iter().any(|&b| b != 0) {
            return Err(CodecError::Unrepresentable);
        }
        out.extend_from_slice(&bytes[..width]);
    }
    Ok(out)
}

/// Decode exactly `count` unsigned values of `width` bytes.
///
/// When `modulus` is given, values outside `[0, modulus)` are rejected.  A
/// fixed-width field is wider than the modulus, so without this a decoder
/// accepts non-canonical representatives: they re-encode to different
/// bytes and break the "decode then encode is the identity" property a
/// wire format needs.
///
/// The length is required to be exact.  Checking only for truncation let
/// `pk_decode(valid || junk)` succeed and return the original key, so two
/// distinct byte strings decoded to one object — the same malleability the
/// modulus check exists to prevent.
pub fn unpack_unsigned(
    data: &[u8],
    width: usize,
    count: usize,
    modulus: Option<u64>,
) -> Result<Vec<u64>> {
    if width > 8 {
        return Err(CodecError::Unrepresentable);
    }
    if data.len() != width * count {
        return Err(CodecError::LengthMismatch);
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let mut buf = [0u8; 8];
        buf[..width].copy_from_slice(&data[i * width..(i + 1) * width]);
        let v = u64::from_le_bytes(buf);
        if let Some(m) = modulus {
            if v >= m {
                return Err(CodecError::NonCanonical);
            }
        }
        out.push(v);
    }
    Ok(out)
}

pub fn pack_signed(values: &[i64], width: usize, bound: i64) -> Result<Vec<u8>> {
    if width > 8 {
        return Err(CodecError::Unrepresentable);
    }
    let mut out = Vec::with_capacity(values.len() * width);
    for &v in values {
        if v < -bound || v > bound {
            return Err(CodecError::OutOfRange);
        }
        // `i128` for the same reason as `Coder::Signed`: the offset can
        // carry `2·bound` past `i64`.
        let bytes = ((v as i128 + bound as i128) as u64).to_le_bytes();
        // Same high-byte check as `pack_unsigned`, and for the same
        // reason: `width` is the caller's, not derived here, and at
        // `bound = 200, width = 1` the offset 400 would pack as 144 and
        // decode as -56.  The reference raises there.
        if bytes[width..].iter().any(|&b| b != 0) {
            return Err(CodecError::Unrepresentable);
        }
        out.extend_from_slice(&bytes[..width]);
    }
    Ok(out)
}

/// Decode exactly `count` signed values; see [`unpack_unsigned`] on why
/// the length is exact.
pub fn unpack_signed(data: &[u8], width: usize, count: usize, bound: i64) -> Result<Vec<i64>> {
    if width > 8 {
        return Err(CodecError::Unrepresentable);
    }
    if data.len() != width * count {
        return Err(CodecError::LengthMismatch);
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let mut buf = [0u8; 8];
        buf[..width].copy_from_slice(&data[i * width..(i + 1) * width]);
        let v = u64::from_le_bytes(buf) as i128 - bound as i128;
        if v < -(bound as i128) || v > bound as i128 {
            return Err(CodecError::OutOfRange);
        }
        out.push(v as i64);
    }
    Ok(out)
}

// ---- polynomial / vector helpers -----------------------------------------

/// Encode one polynomial of exactly `ring.d` coefficients.
///
/// The length check is the point.  Without it a vector of polynomials of
/// lengths `31, 33, 32, …` encodes to the same bytes as a properly
/// partitioned one — the encoder just concatenates — and decoding returns
/// the *other* object.  Two distinct values with one encoding is exactly
/// what a canonical format must not have, and here it would also give two
/// public keys one transcript digest.
pub fn encode_poly_mod(ring: &Ring, poly: &[u64], width: usize) -> Result<Vec<u8>> {
    if poly.len() != ring.d {
        return Err(CodecError::LengthMismatch);
    }
    pack_unsigned(&ring.reduce(poly), width)
}

pub fn decode_poly_mod(ring: &Ring, data: &[u8], width: usize) -> Result<Poly> {
    unpack_unsigned(data, width, ring.d, Some(ring.q))
}

/// Encode `length` polynomials, each of exactly `ring.d` coefficients.
pub fn encode_vec_mod(ring: &Ring, vec: &[Poly], width: usize, length: usize) -> Result<Vec<u8>> {
    if vec.len() != length {
        return Err(CodecError::LengthMismatch);
    }
    let mut out = Vec::with_capacity(vec.len() * ring.d * width);
    for p in vec {
        out.extend_from_slice(&encode_poly_mod(ring, p, width)?);
    }
    Ok(out)
}

pub fn decode_vec_mod(ring: &Ring, data: &[u8], width: usize, length: usize) -> Result<Vec<Poly>> {
    let step = width * ring.d;
    if data.len() != step * length {
        return Err(CodecError::LengthMismatch);
    }
    (0..length)
        .map(|i| decode_poly_mod(ring, &data[i * step..(i + 1) * step], width))
        .collect()
}

/// Encode `length` polynomials by their centred representatives.
pub fn encode_vec_signed(ring: &Ring, vec: &[Poly], bound: i64, length: usize) -> Result<Vec<u8>> {
    if vec.len() != length || vec.iter().any(|p| p.len() != ring.d) {
        return Err(CodecError::LengthMismatch);
    }
    let width = width_for_bound(bound)?;
    let flat = ring.flat_centered(vec);
    pack_signed(&flat, width, bound)
}

pub fn decode_vec_signed(ring: &Ring, data: &[u8], bound: i64, length: usize) -> Result<Vec<Poly>> {
    let width = width_for_bound(bound)?;
    let flat = unpack_signed(data, width, length * ring.d, bound)?;
    Ok((0..length)
        .map(|i| ring.from_centered(&flat[i * ring.d..(i + 1) * ring.d]))
        .collect())
}

/// Encode `rows` rows of exactly `cols` plain integers, already centred.
pub fn encode_int_vec(vec: &[Vec<i64>], bound: i64, rows: usize, cols: usize) -> Result<Vec<u8>> {
    if vec.len() != rows || vec.iter().any(|r| r.len() != cols) {
        return Err(CodecError::LengthMismatch);
    }
    let width = width_for_bound(bound)?;
    let flat: Vec<i64> = vec.iter().flatten().copied().collect();
    pack_signed(&flat, width, bound)
}

pub fn decode_int_vec(data: &[u8], bound: i64, rows: usize, cols: usize) -> Result<Vec<Vec<i64>>> {
    // `chunks(0)` panics, and a zero-column field has no encoding to
    // decode in the first place.
    if cols == 0 {
        return Err(CodecError::Unrepresentable);
    }
    let width = width_for_bound(bound)?;
    let flat = unpack_signed(data, width, rows * cols, bound)?;
    Ok(flat.chunks(cols).map(<[i64]>::to_vec).collect())
}

// ---- the RiVeR codec -----------------------------------------------------

/// Encoders and decoders bound to one parameter profile.
#[derive(Clone)]
pub struct RiVeRCodec {
    pub par: RiVeRParams,
    pub rq: Ring,
    pub rp: Ring,
    pub rqhat: Ring,

    /// Field widths, all derived from the profile.
    pub w_q: usize,
    pub w_p: usize,
    pub w_qhat: usize,

    /// High bits of the selector commitment `B` live in
    /// High bits of the selector commitment `B`.  `[[.]]_K` is taken on
    /// the **centred** representative, so
    /// these are signed and about half of them are negative.
    pub bound_b_hi: i64,
    pub w_b_hi: usize,

    /// Response bounds, exactly the ones the verifier enforces, and
    /// exactly: `floor_sqrt` of the exact squared bound is the largest
    /// integer that can pass, so the encoder's cap and the acceptance
    /// test agree by construction rather than by a `ceil` on a float that
    /// could sit either side of it.
    pub bound_f1: i64,
    pub bound_zb: i64,
    pub bound_zs: i64,
    pub bound_zm: i64,
    pub bound_x: i64,

    /// `pi_OOM = (B, x, f_1, z_b, z)`.  The masked responses are Gaussian
    /// and go through Rice; `B` and `x` are uniform on their ranges and go
    /// through fixed width.
    ///
    /// `z` is split on the wire.  The paper gives its two blocks
    /// different widths — `z_s` at `sigma_s`, `z_m` at `sigma_m`, and
    /// `sigma_m / sigma_s` is between 3.9 and 5.7 across the profiles — so
    /// one Rice parameter for the whole vector would cost roughly a bit
    /// per coefficient on whichever block it did not fit.  They are two
    /// fields with their own parameters and their own bounds;
    /// [`RiVeRCodec::oom_field_values`] is where `z` is split and
    /// [`RiVeRCodec::oom_z_from_values`] where it is reassembled, because
    /// the protocol and the verifier's Euclidean check operate on the
    /// whole vector.
    pub oom_layout: Layout,

    /// The frozen wire manifest for this profile, when it has one.
    ///
    /// Production reads the pinned `(sigma_num, sigma_den, k, bound)`
    /// from here rather than re-deriving them from an `f64` chain; the
    /// re-derivation lives in [`crate::manifest`]'s tests, where a
    /// divergence fails as a named assertion instead of as "proof bytes
    /// differ" three layers up.  `None` only for a profile a caller built
    /// by hand, which then falls back to deriving.
    pub manifest: Option<&'static crate::manifest::ProfileManifest>,
}

impl RiVeRCodec {
    pub fn new(par: RiVeRParams) -> Self {
        let rq = Ring::new(par.q(), par.d);
        let rp = Ring::new(par.p, par.d);
        let rqhat = Ring::new(par.q_hat, par.d);

        let w_q = width_for_modulus(par.q());
        let w_p = width_for_modulus(par.p);
        let w_qhat = width_for_modulus(par.q_hat);

        let manifest = crate::manifest::for_params(&par);

        let bound_b_hi = crate::ring::high_bits_bound(par.q_hat, par.K_b);
        let w_b_hi = width_for_bound(bound_b_hi).expect("B high-bit bound is non-negative");

        let bound_f1 = par.f1_inf_bound_sq().floor_sqrt() as i64;
        let bound_zb = par.zb_inf_bound_sq().floor_sqrt() as i64;
        let bound_zs = par.zs_inf_bound_sq().floor_sqrt() as i64;
        let bound_zm = par.zm_inf_bound_sq().floor_sqrt() as i64;
        let bound_x = par.gamma as i64;

        // Rice parameters come from the frozen manifest when the profile
        // has one, and are derived otherwise.  Both paths agree — the
        // manifest's own tests re-derive every entry — but only the table
        // is independent of the order an `f64` chain is evaluated in.
        let rice = |spec: Option<crate::manifest::GaussianSpec>, sigma: f64, bound: i64| match spec
        {
            Some(g) => {
                debug_assert_eq!(g.bound, bound, "manifest bound disagrees with the profile");
                Coder::rice_with_k(g.rice_k, bound)
            }
            None => Coder::rice_sigma(sigma, bound),
        };

        let oom_layout = Layout::new(vec![
            Field::rows("B", Coder::signed(bound_b_hi), par.d, par.n_hat),
            Field::flat("x", Coder::signed(bound_x), par.d),
            Field::rows(
                "f1",
                rice(manifest.map(|m| m.f1), par.sigma_a(), bound_f1),
                par.d,
                par.N - 1,
            ),
            Field::rows(
                "zb",
                rice(manifest.map(|m| m.zb), par.sigma_b(), bound_zb),
                par.d,
                par.k_hat,
            ),
            Field::ring_rows(
                "zs",
                rice(manifest.map(|m| m.zs), par.sigma_s(), bound_zs),
                par.d,
                par.s_dim(),
                par.q(),
            ),
            Field::ring_rows(
                "zm",
                rice(manifest.map(|m| m.zm), par.sigma_m(), bound_zm),
                par.d,
                par.m_dim(),
                par.q(),
            ),
        ]);

        Self {
            par,
            rq,
            rp,
            rqhat,
            w_q,
            w_p,
            w_qhat,
            bound_b_hi,
            w_b_hi,
            bound_f1,
            bound_zb,
            bound_zs,
            bound_zm,
            bound_x,
            oom_layout,
            manifest,
        }
    }

    // -- public key --------------------------------------------------------

    /// `t` in `R_p^n`.
    ///
    /// Fallible, and the shape check is not a formality: these encoders
    /// concatenate, so `n` polynomials of the wrong individual lengths
    /// can produce the same bytes as `n` correct ones.  Until the scheme
    /// layer exists to validate keys before use, the encoder is where
    /// that is caught — otherwise two distinct keys share a
    /// [`ring_digest`], which is a Fiat–Shamir collision rather than a
    /// serialization nit.
    pub fn pk_encode(&self, pk: &[Poly]) -> Result<Vec<u8>> {
        encode_vec_mod(&self.rp, pk, self.w_p, self.par.n)
    }

    pub fn pk_decode(&self, data: &[u8]) -> Result<Vec<Poly>> {
        decode_vec_mod(&self.rp, data, self.w_p, self.par.n)
    }

    pub fn pk_bytes(&self) -> usize {
        self.w_p * self.par.d * self.par.n
    }

    // -- secret key --------------------------------------------------------

    /// `s` in `S_beta^ell`, one signed byte per coefficient.
    pub fn sk_encode(&self, sk: &[Poly]) -> Result<Vec<u8>> {
        encode_vec_signed(&self.rq, sk, self.par.beta as i64, self.par.ell)
    }

    pub fn sk_decode(&self, data: &[u8]) -> Result<Vec<Poly>> {
        decode_vec_signed(&self.rq, data, self.par.beta as i64, self.par.ell)
    }

    // -- VRF value ---------------------------------------------------------

    pub fn value_encode(&self, v: &[u64]) -> Result<Vec<u8>> {
        encode_poly_mod(&self.rp, v, self.w_p)
    }

    pub fn value_decode(&self, data: &[u8]) -> Result<Poly> {
        decode_poly_mod(&self.rp, data, self.w_p)
    }

    // -- challenge ---------------------------------------------------------

    pub fn challenge_encode(&self, x: &[i64]) -> Result<Vec<u8>> {
        encode_int_vec(&[x.to_vec()], self.bound_x, 1, self.par.d)
    }

    pub fn challenge_decode(&self, data: &[u8]) -> Result<Vec<i64>> {
        let rows = decode_int_vec(data, self.bound_x, 1, self.par.d)?;
        Ok(rows.into_iter().next().unwrap())
    }

    // -- OOM proof ---------------------------------------------------------
    // `B`, `x`, `f_1` and `z^bin` are carried as *integer* polynomials —
    // the selector layer needs their exact integer values — so they encode
    // and decode as signed integers.  `z` is a genuine `R_q` element.

    /// `pi_OOM = (B, x, f_1, z_b, z)`, positional in layout order.
    pub fn oom_encode(&self, pi: &[FieldValue]) -> Result<Vec<u8>> {
        self.oom_layout.encode(pi)
    }

    /// The six layout values for one OOM proof, splitting `z` into its
    /// `z_s` and `z_m` blocks.
    ///
    /// The split lives here rather than at the call sites so the wire
    /// order and the block boundary are stated once.  `Err` if `z` is not
    /// `r_dim` rows — a caller can only reach that with a proof it built
    /// itself, and encoding a short `z` would silently move the boundary.
    pub fn oom_field_values(
        &self,
        b_hi: &[Vec<i64>],
        x: &[i64],
        f1: &[Vec<i64>],
        zb: &[Vec<i64>],
        z: &[Vec<u64>],
    ) -> Result<Vec<FieldValue>> {
        if z.len() != self.par.r_dim() {
            return Err(CodecError::LengthMismatch);
        }
        let (zs, zm) = z.split_at(self.par.s_dim());
        Ok(vec![
            FieldValue::Ints(b_hi.to_vec()),
            FieldValue::flat(x.to_vec()),
            FieldValue::Ints(f1.to_vec()),
            FieldValue::Ints(zb.to_vec()),
            FieldValue::Residues(zs.to_vec()),
            FieldValue::Residues(zm.to_vec()),
        ])
    }

    /// Reassemble `z` from the decoded `z_s` and `z_m` blocks.
    ///
    /// The protocol and the verifier's Euclidean check operate on the
    /// whole vector, so the two halves come back together immediately
    /// after decoding.
    pub fn oom_z_from_values(&self, zs: Vec<Vec<u64>>, zm: Vec<Vec<u64>>) -> Vec<Vec<u64>> {
        let mut z = zs;
        z.extend(zm);
        z
    }

    pub fn oom_decode(&self, data: &[u8]) -> Result<Vec<FieldValue>> {
        self.oom_layout.decode(data)
    }

    /// Worst-case `|pi_OOM|`; the actual length varies with the Gaussians.
    ///
    /// The reference's `proof_sizes` reporting helper has no counterpart
    /// here: it exists to print measured sizes against the paper's
    /// estimate, which belongs with the bench harness rather than the wire
    /// format.  This and [`Layout::min_bytes`] are the parts the codec
    /// itself needs.
    pub fn oom_max_bytes(&self) -> usize {
        self.oom_layout.max_bytes()
    }

    // -- full proof --------------------------------------------------------

    /// Frame `pi = (pi_OOM, pi_ex)` as two length-prefixed blocks.
    ///
    /// Both are prefixed because with Rice coding neither has a length the
    /// reader can compute in advance, and a self-delimiting format that
    /// guessed would be one more thing an attacker could steer.
    ///
    /// The exact block arrives already encoded: its layout belongs to
    /// whichever exact backend is in use, which this module does not need
    /// to know.
    pub fn proof_encode(&self, oom: &[FieldValue], ex_bytes: &[u8]) -> Result<Vec<u8>> {
        let oom_bytes = self.oom_encode(oom)?;
        Ok(proof_frame(&oom_bytes, ex_bytes))
    }

    /// Split a framed proof, bounding each block by its own layout.
    pub fn proof_split<'a>(
        &self,
        data: &'a [u8],
        ex_layout: &Layout,
    ) -> Result<(&'a [u8], &'a [u8])> {
        proof_unframe(data, &self.oom_layout, ex_layout)
    }
}

/// Concatenate two 4-byte-length-prefixed blocks.
pub fn proof_frame(oom: &[u8], ex: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + oom.len() + ex.len());
    out.extend_from_slice(&(oom.len() as u32).to_le_bytes());
    out.extend_from_slice(oom);
    out.extend_from_slice(&(ex.len() as u32).to_le_bytes());
    out.extend_from_slice(ex);
    out
}

/// Inverse of [`proof_frame`], with both prefixes validated.
///
/// A prefix is attacker-controlled, so it is checked against what the
/// profile can actually produce before any slicing: a claim of
/// `0xFFFFFFFF` must not become a 4 GB read, and a claim below
/// `min_bytes` cannot be a well-formed block either.
pub fn proof_unframe<'a>(
    data: &'a [u8],
    oom_layout: &Layout,
    ex_layout: &Layout,
) -> Result<(&'a [u8], &'a [u8])> {
    fn take<'b>(data: &'b [u8], off: usize, layout: &Layout) -> Result<(&'b [u8], usize)> {
        if data.len() < off + 4 {
            return Err(CodecError::Truncated);
        }
        let n = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        if n < layout.min_bytes() || n > layout.max_bytes() {
            return Err(CodecError::BadLengthPrefix);
        }
        if data.len() < off + 4 + n {
            return Err(CodecError::Truncated);
        }
        Ok((&data[off + 4..off + 4 + n], off + 4 + n))
    }

    let (oom, off) = take(data, 0, oom_layout)?;
    let (ex, off) = take(data, off, ex_layout)?;
    if off != data.len() {
        return Err(CodecError::TrailingBytes);
    }
    Ok((oom, ex))
}

// ---- Fiat–Shamir transcript ----------------------------------------------

/// Digest of the ring and value part of `rho' = (R~, v, W)`.
///
/// `W` is appended separately by the caller because its encoding belongs
/// to whichever exact backend is in use.
/// Fallible because its inputs are: a ring whose keys are the wrong shape
/// has no well-defined digest, and returning one anyway is how two keys
/// come to share a challenge.
pub fn ring_digest(codec: &RiVeRCodec, ring_pks: &[Vec<Poly>], value: &[u64]) -> Result<Vec<u8>> {
    let mut blobs: Vec<Vec<u8>> = ring_pks
        .iter()
        .map(|t| codec.pk_encode(t))
        .collect::<Result<_>>()?;
    blobs.push(codec.value_encode(value)?);
    let parts: Vec<Part<'_>> = blobs.iter().map(|b| Part::Bytes(b.as_slice())).collect();
    Ok(hash_bytes(32, &[DS_CHALLENGE, b".rho"].concat(), &parts))
}

/// Digest standing for `ck_{r,m}` in the Fiat–Shamir input.
///
/// `ck_{r,m} = [A | -I_n | 0 ; h_m^T | 0 | -1]` and `A = SamMat(rho, ...)`,
/// so the pair `(rho, h_m)` determines the whole matrix.  Hashing that
/// pair is an injective stand-in for hashing the matrix, and avoids
/// serializing `n x ell` ring elements on every call.
pub fn statement_digest(codec: &RiVeRCodec, seed: &[u8], h_m: &[Poly]) -> Result<Vec<u8>> {
    if h_m.len() != codec.par.ell {
        return Err(CodecError::LengthMismatch);
    }
    let blobs: Vec<Vec<u8>> = h_m
        .iter()
        .map(|p| encode_poly_mod(&codec.rq, p, codec.w_q))
        .collect::<Result<_>>()?;
    let mut parts: Vec<Part<'_>> = Vec::with_capacity(1 + blobs.len());
    parts.push(Part::Bytes(seed));
    parts.extend(blobs.iter().map(|b| Part::Bytes(b.as_slice())));
    Ok(hash_bytes(32, &[DS_CHALLENGE, b".ck"].concat(), &parts))
}

// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::RIVER_TOY;
    use crate::sample::{gaussian_int, Xof, GAUSSIAN_TAILCUT};

    fn lcg(seed: u64) -> impl FnMut(u64) -> u64 {
        let mut s = seed | 1;
        move |n| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if n == 0 {
                0
            } else {
                (s >> 11) % n
            }
        }
    }

    /// A structurally valid `pi_OOM` at the TOY profile, deterministic in
    /// `seed`.  Every coder sees a value inside its range, and the first
    /// row of each field is pinned to the extremes so the worst-case unary
    /// run and both signs are exercised.
    fn sample_oom(codec: &RiVeRCodec, seed: u64) -> Vec<FieldValue> {
        let par = &codec.par;
        let mut rand = lcg(seed);
        let mut out = Vec::new();

        // `B` is signed took `[[.]]_K` on the
        // centred representative, so the extremes are both ends.
        let mut b = Vec::new();
        for i in 0..par.n_hat {
            let row: Vec<i64> = (0..par.d)
                .map(|j| {
                    if i == 0 && j < 3 {
                        [0, codec.bound_b_hi, -codec.bound_b_hi][j]
                    } else {
                        rand(2 * codec.bound_b_hi as u64 + 1) as i64 - codec.bound_b_hi
                    }
                })
                .collect();
            b.push(row);
        }
        out.push(FieldValue::Ints(b));

        let x: Vec<i64> = (0..par.d)
            .map(|j| match j {
                0 => 0,
                1 => codec.bound_x,
                2 => -codec.bound_x,
                _ => rand(2 * codec.bound_x as u64 + 1) as i64 - codec.bound_x,
            })
            .collect();
        out.push(FieldValue::flat(x));

        let mut gauss_field = |rows: usize, bound: i64| {
            let mut v = Vec::new();
            for i in 0..rows {
                let row: Vec<i64> = (0..par.d)
                    .map(|j| {
                        if i == 0 && j < 3 {
                            [0, bound, -bound][j]
                        } else {
                            rand(2 * bound as u64 + 1) as i64 - bound
                        }
                    })
                    .collect();
                v.push(row);
            }
            v
        };
        out.push(FieldValue::Ints(gauss_field(par.N - 1, codec.bound_f1)));
        out.push(FieldValue::Ints(gauss_field(par.k_hat, codec.bound_zb)));

        // `z` is two fields on the wire, at two widths — see
        // `RiVeRCodec::oom_field_values`.
        let q = par.q();
        let residues = |rows: Vec<Vec<i64>>| -> Vec<Poly> {
            rows.into_iter()
                .map(|row| row.iter().map(|&c| c.rem_euclid(q as i64) as u64).collect())
                .collect()
        };
        out.push(FieldValue::Residues(residues(gauss_field(
            par.s_dim(),
            codec.bound_zs,
        ))));
        out.push(FieldValue::Residues(residues(gauss_field(
            par.m_dim(),
            codec.bound_zm,
        ))));

        out
    }

    // ---- primitives ------------------------------------------------------

    #[test]
    fn width_helpers_match_the_reference() {
        assert_eq!(width_for_modulus(255), 1);
        assert_eq!(width_for_modulus(256), 2);
        assert_eq!(width_for_bound(127).unwrap(), 1); // [-127,127] -> 255 values
        assert_eq!(width_for_bound(128).unwrap(), 2);
    }

    #[test]
    fn unsigned_round_trip() {
        let values = vec![0u64, 1, 65535];
        let blob = pack_unsigned(&values, 2).unwrap();
        assert_eq!(unpack_unsigned(&blob, 2, 3, None).unwrap(), values);
        // a value too wide for the field is an error, not a truncation:
        // 65536 would otherwise pack as 0 and re-encode differently
        assert_eq!(pack_unsigned(&[65536], 2), Err(CodecError::Unrepresentable));
        // and a width past u64 is reported, not panicked
        for r in [
            pack_unsigned(&[0], 9).err(),
            unpack_unsigned(&[0; 9], 9, 1, None).err(),
            pack_signed(&[0], 9, 1).err(),
            unpack_signed(&[0; 9], 9, 1, 1).err(),
        ] {
            assert_eq!(r, Some(CodecError::Unrepresentable));
        }
    }

    #[test]
    fn signed_round_trip_and_bound_enforcement() {
        let values = vec![-100i64, 0, 100];
        let blob = pack_signed(&values, 2, 100).unwrap();
        assert_eq!(unpack_signed(&blob, 2, 3, 100).unwrap(), values);
        for bad in [-101i64, 101] {
            assert_eq!(pack_signed(&[bad], 2, 100), Err(CodecError::OutOfRange));
        }
    }

    #[test]
    fn corrupted_signed_block_does_not_decode_out_of_range() {
        let blob = pack_signed(&[100], 2, 100).unwrap();
        let corrupted = 0xFFFFu16.to_le_bytes();
        assert_ne!(blob.as_slice(), corrupted.as_slice());
        assert_eq!(
            unpack_signed(&corrupted, 2, 1, 100),
            Err(CodecError::OutOfRange)
        );
    }

    #[test]
    fn truncated_and_overlong_blocks_are_rejected() {
        assert_eq!(
            unpack_unsigned(&[0], 2, 1, None),
            Err(CodecError::LengthMismatch)
        );
        assert_eq!(
            unpack_unsigned(&[0, 0, 0], 2, 1, None),
            Err(CodecError::LengthMismatch)
        );
    }

    #[test]
    fn int_vec_round_trip() {
        let mut rand = lcg(1);
        let vec: Vec<Vec<i64>> = (0..3)
            .map(|_| (0..8).map(|_| rand(101) as i64 - 50).collect())
            .collect();
        let blob = encode_int_vec(&vec, 50, 3, 8).unwrap();
        assert_eq!(decode_int_vec(&blob, 50, 3, 8).unwrap(), vec);
        // and a ragged or mis-sized container is refused, not concatenated
        let ragged = vec![vec![0i64; 7], vec![0i64; 9], vec![0i64; 8]];
        assert_eq!(
            encode_int_vec(&ragged, 50, 3, 8),
            Err(CodecError::LengthMismatch)
        );
    }

    // ---- bit-level codec -------------------------------------------------

    #[test]
    fn bit_writer_reader_round_trip() {
        let mut rand = lcg(11);
        for _ in 0..200 {
            let n = 1 + rand(19) as usize;
            let widths: Vec<u32> = (0..n).map(|_| 1 + rand(32) as u32).collect();
            let values: Vec<u64> = widths
                .iter()
                .map(|&w| rand(1u64 << w.min(63)) & ((1u64 << w) - 1))
                .collect();
            let mut w = BitWriter::new();
            for (&value, &width) in values.iter().zip(&widths) {
                w.write_bits(value, width);
            }
            let blob = w.to_bytes();
            let mut r = BitReader::new(&blob);
            let got: Vec<u64> = widths.iter().map(|&w| r.read_bits(w).unwrap()).collect();
            assert_eq!(got, values);
        }
    }

    #[test]
    fn wide_fields_survive_the_chunked_path() {
        // The `> 56` split in `write_bits` / `read_bits` has no coverage
        // from any real layout — the widest is 54 bits — so pin it here.
        for width in [56u32, 57, 63, 64] {
            let value = 0x0123_4567_89AB_CDEFu64 >> (64 - width);
            let mut w = BitWriter::new();
            w.write_bits(value, width);
            w.write_bits(1, 3);
            let blob = w.to_bytes();
            let mut r = BitReader::new(&blob);
            assert_eq!(r.read_bits(width).unwrap(), value, "width {width}");
            assert_eq!(r.read_bits(3).unwrap(), 1);
        }
    }

    #[test]
    fn a_signed_field_at_the_width_limit_does_not_overflow() {
        // No profile reaches a bound past `2^62` — `gamma` is 16 — but a
        // coder is constructible with one, and then `value + bound` leaves
        // `i64`.  Both directions carry the offset through `i128`, so this
        // is a round trip rather than a debug-build panic.
        let bound = (1i64 << 62) + 12345;
        let coder = Coder::signed(bound);
        assert_eq!(coder.max_bits(), 64);
        let values = [0i64, 1, -1, bound, -bound];
        let mut w = BitWriter::new();
        for &v in &values {
            coder.write(&mut w, v).unwrap();
        }
        let blob = w.to_bytes();
        let mut r = BitReader::new(&blob);
        let got: Vec<i64> = values.iter().map(|_| coder.read(&mut r).unwrap()).collect();
        assert_eq!(got, values);

        // and hostile bytes are out of range, not a wrapped small value
        assert_eq!(
            coder.read(&mut BitReader::new(&[0xFFu8; 8])),
            Err(CodecError::OutOfRange)
        );
        let width = width_for_bound(bound).unwrap();
        assert_eq!(width, 8);
        assert_eq!(
            unpack_signed(&[0xFFu8; 8], width, 1, bound),
            Err(CodecError::OutOfRange)
        );
        let packed = pack_signed(&values, width, bound).unwrap();
        assert_eq!(
            unpack_signed(&packed, width, values.len(), bound).unwrap(),
            values
        );
    }

    #[test]
    fn unary_round_trip_including_long_runs() {
        for value in [0u64, 1, 7, 31, 32, 33, 100, 255] {
            let mut w = BitWriter::new();
            w.write_unary(value);
            let blob = w.to_bytes();
            assert_eq!(BitReader::new(&blob).read_unary(300).unwrap(), value);
        }
    }

    #[test]
    fn rice_round_trip_over_the_whole_range() {
        let coder = Coder::rice_sigma(352.0, 2112);
        let mut values: Vec<i64> = (-2112..=2112).step_by(7).collect();
        values.extend([0, 1, -1, 2112, -2112]);
        let mut w = BitWriter::new();
        for &v in &values {
            coder.write(&mut w, v).unwrap();
        }
        let blob = w.to_bytes();
        let mut r = BitReader::new(&blob);
        let got: Vec<i64> = values.iter().map(|_| coder.read(&mut r).unwrap()).collect();
        assert_eq!(got, values);
    }

    #[test]
    fn rice_beats_fixed_width_on_a_gaussian() {
        // The point of the exercise: Rice must actually pay for itself.
        let coder = Coder::rice_sigma(352.0, 2112);
        let (num, den) = rational_sigma(352.0);
        let mut x = Xof::new(b"codec-test", &[Part::Bytes(b"rice")]);
        let mut w = BitWriter::new();
        let count = 4000;
        for _ in 0..count {
            coder
                .write(&mut w, gaussian_int(&mut x, num, den, GAUSSIAN_TAILCUT))
                .unwrap();
        }
        let bits = w.bit_length() as f64 / count as f64;
        assert!((10.0..12.0).contains(&bits), "{bits} bits per coefficient");
        assert!(bits < Coder::signed(2112).max_bits() as f64);
    }

    #[test]
    fn rice_max_bits_is_exactly_reached_by_a_coefficient_at_the_bound() {
        // The direct check the old `max_bits` failed: measure the worst
        // case instead of deriving it from the DoS cap.  It was one bit
        // per coefficient too large, which no round-trip test could see
        // because `max_bits` never touches an encoding.
        for (sigma, bound) in [(352.0, 4970i64), (352.0, 2112), (8.0, 100), (1.0, 3)] {
            let coder = Coder::rice_sigma(sigma, bound);
            let mut w = BitWriter::new();
            coder.write(&mut w, -bound).unwrap();
            assert_eq!(
                w.bit_length(),
                coder.max_bits(),
                "sigma {sigma} bound {bound}: worst case is not the declared maximum"
            );
            for v in [0, 1, -1, bound / 2, bound - 1, bound] {
                let mut w2 = BitWriter::new();
                coder.write(&mut w2, v).unwrap();
                assert!(w2.bit_length() <= coder.max_bits(), "{v} exceeds max_bits");
                assert!(w2.bit_length() >= coder.min_bits(), "{v} below min_bits");
            }
        }
    }

    #[test]
    fn layout_bounds_bracket_a_real_encoding() {
        let codec = RiVeRCodec::new(RIVER_TOY);
        let blob = codec.oom_encode(&sample_oom(&codec, 21)).unwrap();
        assert!(blob.len() <= codec.oom_layout.max_bytes());
        assert!(blob.len() >= codec.oom_layout.min_bytes());
    }

    #[test]
    fn degenerate_configurations_are_errors_and_never_panics() {
        // The generic API edges, none of which any profile reaches — but
        // "total for any argument" is the claim, so each is pinned.
        // Every one of these aborted or silently corrupted before.

        // a zero bound: the only encodable value is 0, whose sign is
        // omitted, so the worst case is k + 1 and not k + 2
        for k in [0u32, 1, 5] {
            let coder = Coder::rice_with_k(k, 0);
            let mut w = BitWriter::new();
            coder.write(&mut w, 0).unwrap();
            assert_eq!(w.bit_length(), coder.max_bits(), "Rice(k={k}, bound=0)");
            assert_eq!(coder.max_bits(), coder.min_bits());
            let blob = w.to_bytes();
            assert_eq!(coder.read(&mut BitReader::new(&blob)).unwrap(), 0);
        }

        // an undersized signed field truncated: 200 + 200 = 400 packed as
        // 144 and decoded as -56
        assert_eq!(
            pack_signed(&[200], 1, 200),
            Err(CodecError::Unrepresentable)
        );
        assert!(pack_signed(&[200], 2, 200).is_ok());

        // a negative bound names no range
        assert_eq!(width_for_bound(-1), Err(CodecError::Unrepresentable));
        assert_eq!(width_for_bound(0).unwrap(), 1);

        // a zero-column field: `chunks(0)` panicked
        assert_eq!(
            decode_int_vec(&[], 1, 0, 0),
            Err(CodecError::Unrepresentable)
        );

        // a zero ring modulus: encode refused only by accident, decode
        // divided by zero
        let zero_ring = Layout::new(vec![Field::ring_rows("z", Coder::signed(4), 2, 1, 0)]);
        assert_eq!(
            zero_ring.encode(&[FieldValue::Residues(vec![vec![0u64; 2]])]),
            Err(CodecError::Unrepresentable)
        );
        assert_eq!(zero_ring.decode(&[0u8]), Err(CodecError::Unrepresentable));

        // a `Coder` built past what the constructors allow — the variants
        // are public data, so the use sites check too
        let bad = Coder::Rice {
            k: 70,
            bound: 8,
            max_high: 1,
        };
        assert_eq!(
            bad.write(&mut BitWriter::new(), 1),
            Err(CodecError::Unrepresentable)
        );
        assert_eq!(
            bad.read(&mut BitReader::new(&[0xFF; 16])),
            Err(CodecError::Unrepresentable)
        );
    }

    #[test]
    fn rice_rejects_out_of_bound_values() {
        let coder = Coder::rice_sigma(352.0, 2112);
        for bad in [2113i64, -2113, 1_000_000_000] {
            assert_eq!(
                coder.write(&mut BitWriter::new(), bad),
                Err(CodecError::OutOfRange)
            );
        }
    }

    #[test]
    fn optimal_rice_k_is_integer_deterministic() {
        // `k` decides the wire format, so it must not depend on float
        // rounding anywhere.
        assert_eq!(optimal_rice_k_f(352.0), 8);
        assert_eq!(optimal_rice_k_f(4096.0), 12);
        assert_eq!(optimal_rice_k_f(1.5e7), 24);
        assert_eq!(optimal_rice_k_f(0.5), 0);
        for sigma in [1.0, 2.0, 3.0, 100.0, 1e6] {
            let (num, den) = rational_sigma(sigma);
            assert_eq!(optimal_rice_k_f(sigma), optimal_rice_k(num, den));
        }
    }

    #[test]
    fn rice_encoding_matches_the_pinned_blob() {
        // The same vector `river-py/test_kat.py` layer 4 pins.
        let coder = Coder::rice_sigma(352.0, 4970);
        let mut w = BitWriter::new();
        for v in [0i64, 1, -1, 255, -256, 4970, -4970] {
            coder.write(&mut w, v).unwrap();
        }
        assert_eq!(hex::encode(w.to_bytes()), "000208f01f80aafdff1fb5ffff0b");
    }

    #[test]
    fn uniform_rejects_non_canonical_residues() {
        let coder = Coder::uniform(61);
        assert_eq!(coder.max_bits(), 6); // 6 bits hold 0..63
        let mut w = BitWriter::new();
        w.write_bits(62, 6); // a value the modulus excludes
        let blob = w.to_bytes();
        assert_eq!(
            coder.read(&mut BitReader::new(&blob)),
            Err(CodecError::NonCanonical)
        );
    }

    #[test]
    fn pinned_field_widths() {
        assert_eq!(Coder::uniform(61).max_bits(), 6);
        assert_eq!(Coder::uniform(67112897).max_bits(), 27); // the pre-1-Aug q~
        assert_eq!(Coder::uniform(427634113).max_bits(), 29); // q~, below 2^29
        assert_eq!(Coder::signed(1).max_bits(), 2); // ternary
        assert_eq!(Coder::signed(16).max_bits(), 6); // the OOM challenge range
    }

    // ---- scheme objects --------------------------------------------------

    #[test]
    fn pk_round_trip_and_size() {
        let codec = RiVeRCodec::new(RIVER_TOY);
        let mut rand = lcg(2);
        let pk: Vec<Poly> = (0..codec.par.n)
            .map(|_| (0..codec.par.d).map(|_| rand(codec.par.p)).collect())
            .collect();
        let blob = codec.pk_encode(&pk).unwrap();
        assert_eq!(blob.len(), codec.pk_bytes());
        assert_eq!(codec.pk_decode(&blob).unwrap(), pk);
        // and the encoding is canonical
        assert_eq!(
            codec.pk_encode(&codec.pk_decode(&blob).unwrap()).unwrap(),
            blob
        );
    }

    #[test]
    fn misshapen_scheme_objects_cannot_collide_with_well_formed_ones() {
        // These encoders concatenate, so before the shape checks a key
        // whose polynomials were 31, 33, 32, … coefficients long encoded
        // to exactly the bytes of a properly partitioned one — and
        // decoding returned the *other* key.  Two distinct objects with
        // one encoding is one transcript digest for both, which is a
        // Fiat–Shamir collision, not a serialization nit.
        let codec = RiVeRCodec::new(RIVER_TOY);
        let d = codec.par.d;
        let mut rand = lcg(31);
        let good: Vec<Poly> = (0..codec.par.n)
            .map(|_| (0..d).map(|_| rand(codec.par.p)).collect())
            .collect();
        let flat: Vec<u64> = good.iter().flatten().copied().collect();

        // same coefficients, same order, different partition
        let mut ragged: Vec<Poly> = Vec::new();
        let mut at = 0;
        for len in [d - 1, d + 1] {
            ragged.push(flat[at..at + len].to_vec());
            at += len;
        }
        while at < flat.len() {
            ragged.push(flat[at..at + d].to_vec());
            at += d;
        }
        assert_eq!(ragged.len(), good.len());
        assert_eq!(
            codec.pk_encode(&ragged),
            Err(CodecError::LengthMismatch),
            "a ragged key still encodes"
        );
        assert!(ring_digest(&codec, &[ragged], &vec![0; d]).is_err());

        // wrong vector length, right polynomial lengths
        assert_eq!(
            codec.pk_encode(&good[..good.len() - 1]),
            Err(CodecError::LengthMismatch)
        );
        assert_eq!(
            codec.value_encode(&vec![0u64; d - 1]),
            Err(CodecError::LengthMismatch)
        );
        assert_eq!(
            codec.challenge_encode(&vec![0i64; d + 1]),
            Err(CodecError::LengthMismatch)
        );
        assert_eq!(
            codec.sk_encode(&vec![vec![0u64; d]; codec.par.ell + 1]),
            Err(CodecError::LengthMismatch)
        );
        assert_eq!(
            statement_digest(&codec, b"s", &vec![vec![0u64; d]; codec.par.ell - 1]),
            Err(CodecError::LengthMismatch)
        );
        // the well-formed one still works
        assert!(codec.pk_encode(&good).is_ok());
    }

    #[test]
    fn pk_decode_rejects_a_non_canonical_coefficient() {
        // A fixed-width field is wider than the modulus; decoding must not
        // accept a representative outside `[0, p)`.
        let codec = RiVeRCodec::new(RIVER_TOY);
        let pk: Vec<Poly> = (0..codec.par.n)
            .map(|_| vec![codec.par.p - 1; codec.par.d])
            .collect();
        let mut blob = codec.pk_encode(&pk).unwrap();
        let w = codec.w_p;
        blob[..w].copy_from_slice(&(codec.par.p + 1).to_le_bytes()[..w]);
        assert_eq!(codec.pk_decode(&blob), Err(CodecError::NonCanonical));
    }

    #[test]
    fn sk_and_value_round_trip() {
        let codec = RiVeRCodec::new(RIVER_TOY);
        let mut rand = lcg(4);
        let q = codec.par.q();
        let sk: Vec<Poly> = (0..codec.par.ell)
            .map(|_| {
                (0..codec.par.d)
                    .map(|_| [0, 1, q - 1][rand(3) as usize])
                    .collect()
            })
            .collect();
        assert_eq!(codec.sk_decode(&codec.sk_encode(&sk).unwrap()).unwrap(), sk);

        let v: Poly = (0..codec.par.d).map(|_| rand(codec.par.p)).collect();
        assert_eq!(
            codec
                .value_decode(&codec.value_encode(&v).unwrap())
                .unwrap(),
            v
        );
    }

    #[test]
    fn challenge_round_trip() {
        let codec = RiVeRCodec::new(RIVER_TOY);
        let mut rand = lcg(5);
        let g = codec.bound_x;
        let x: Vec<i64> = (0..codec.par.d)
            .map(|_| rand(2 * g as u64 + 1) as i64 - g)
            .collect();
        let blob = codec.challenge_encode(&x).unwrap();
        assert_eq!(codec.challenge_decode(&blob).unwrap(), x);
    }

    // ---- layouts ---------------------------------------------------------

    #[test]
    fn oom_layout_round_trips_and_stays_inside_its_bounds() {
        let codec = RiVeRCodec::new(RIVER_TOY);
        let pi = sample_oom(&codec, 7);
        let blob = codec.oom_encode(&pi).unwrap();
        assert!(blob.len() <= codec.oom_max_bytes());
        assert!(blob.len() >= codec.oom_layout.min_bytes());
        let decoded = codec.oom_decode(&blob).unwrap();
        assert_eq!(decoded, pi);
        assert_eq!(codec.oom_encode(&decoded).unwrap(), blob);
    }

    #[test]
    fn oom_encode_rejects_a_non_canonical_residue() {
        let codec = RiVeRCodec::new(RIVER_TOY);
        let mut pi = sample_oom(&codec, 8);
        let q = codec.par.q();
        match &mut pi[4] {
            FieldValue::Residues(rows) => rows[0][0] = q,
            _ => unreachable!(),
        }
        assert_eq!(codec.oom_encode(&pi), Err(CodecError::NonCanonical));
    }

    #[test]
    fn oom_encode_rejects_a_mismatched_shape() {
        let codec = RiVeRCodec::new(RIVER_TOY);
        let mut pi = sample_oom(&codec, 9);
        match &mut pi[1] {
            FieldValue::Ints(rows) => rows[0].pop(),
            _ => unreachable!(),
        };
        assert_eq!(codec.oom_encode(&pi), Err(CodecError::LengthMismatch));

        // and a value fed to the wrong variant is a mismatch, not a
        // silent reinterpretation of residues as signed integers
        let mut pi = sample_oom(&codec, 9);
        pi[4] = FieldValue::Ints(vec![vec![0; codec.par.d]; codec.par.r_dim()]);
        assert_eq!(codec.oom_encode(&pi), Err(CodecError::LengthMismatch));
    }

    #[test]
    fn decode_rejects_truncation_at_every_prefix() {
        let codec = RiVeRCodec::new(RIVER_TOY);
        let blob = codec.oom_encode(&sample_oom(&codec, 10)).unwrap();
        let step = (blob.len() / 32).max(1);
        for cut in (0..blob.len()).step_by(step) {
            assert!(
                codec.oom_decode(&blob[..cut]).is_err(),
                "accepted a {cut}-byte prefix"
            );
        }
    }

    #[test]
    fn decode_rejects_trailing_bytes() {
        let codec = RiVeRCodec::new(RIVER_TOY);
        let mut blob = codec.oom_encode(&sample_oom(&codec, 11)).unwrap();
        blob.push(0);
        assert_eq!(codec.oom_decode(&blob), Err(CodecError::TrailingBytes));
    }

    #[test]
    fn decode_is_total_and_canonical_on_hostile_input() {
        // Random, mutated and adversarial bytes.  The contract is that
        // *any* malformed encoding surfaces as a `CodecError`, never a
        // panic or an unbounded loop — the all-ones case is the one Rice
        // invites: without the unary cap it spins to the end of the
        // buffer.  Anything that does decode must re-encode to itself.
        let codec = RiVeRCodec::new(RIVER_TOY);
        let blob = codec.oom_encode(&sample_oom(&codec, 12)).unwrap();
        let mut rand = lcg(99);

        let mut cases: Vec<Vec<u8>> = (0..20)
            .map(|_| (0..blob.len()).map(|_| rand(256) as u8).collect())
            .collect();
        cases.push(vec![0xFF; blob.len()]);
        cases.push(vec![0x00; blob.len()]);
        cases.push(Vec::new());
        cases.push(vec![0xFF; 4 * blob.len()]);
        for _ in 0..40 {
            let mut mangled = blob.clone();
            let i = rand(mangled.len() as u64) as usize;
            mangled[i] ^= 1 << rand(8);
            cases.push(mangled);
        }

        for case in cases {
            if let Ok(decoded) = codec.oom_decode(&case) {
                assert_eq!(
                    codec.oom_encode(&decoded).unwrap(),
                    case,
                    "decoded non-canonically"
                );
            }
        }
    }

    #[test]
    fn proof_framing_rejects_hostile_length_prefixes() {
        // Both prefixes are attacker-controlled. One claiming more bytes
        // than exist must not read past the buffer; one claiming fewer
        // must not leave the remainder unaccounted for.
        let codec = RiVeRCodec::new(RIVER_TOY);
        let oom = codec.oom_encode(&sample_oom(&codec, 13)).unwrap();
        // stand-in for an exact-layer block until that layer lands
        let ex_layout = Layout::new(vec![Field::rows(
            "W",
            Coder::uniform(codec.par.q_hat),
            codec.par.d,
            2,
        )]);
        let ex = ex_layout
            .encode(&[FieldValue::Ints(vec![vec![1; codec.par.d]; 2])])
            .unwrap();
        let blob = proof_frame(&oom, &ex);

        let (a, b) = proof_unframe(&blob, &codec.oom_layout, &ex_layout).unwrap();
        assert_eq!((a, b), (oom.as_slice(), ex.as_slice()));

        let n_oom = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
        for at in [0usize, 4 + n_oom] {
            for claim in [0xFFFF_FFFFu32, 0, 1, n_oom as u32 + 1] {
                let mut mangled = blob.clone();
                mangled[at..at + 4].copy_from_slice(&claim.to_le_bytes());
                assert!(
                    proof_unframe(&mangled, &codec.oom_layout, &ex_layout).is_err(),
                    "accepted length {claim} at offset {at}"
                );
            }
        }
        assert!(proof_unframe(&blob[..blob.len() - 1], &codec.oom_layout, &ex_layout).is_err());
    }

    // ---- transcript digests ----------------------------------------------

    #[test]
    fn digests_are_deterministic_and_sensitive() {
        let codec = RiVeRCodec::new(RIVER_TOY);
        let mut rand = lcg(6);
        let pks: Vec<Vec<Poly>> = (0..2)
            .map(|_| {
                (0..codec.par.n)
                    .map(|_| (0..codec.par.d).map(|_| rand(codec.par.p)).collect())
                    .collect()
            })
            .collect();
        let v: Poly = (0..codec.par.d).map(|_| rand(codec.par.p)).collect();

        let a = ring_digest(&codec, &pks, &v).unwrap();
        assert_eq!(a, ring_digest(&codec, &pks, &v).unwrap());
        let swapped: Vec<Vec<Poly>> = pks.iter().rev().cloned().collect();
        assert_ne!(a, ring_digest(&codec, &swapped, &v).unwrap());
        let mut v2 = v.clone();
        v2[0] = (v2[0] + 1) % codec.par.p;
        assert_ne!(a, ring_digest(&codec, &pks, &v2).unwrap());
    }

    #[test]
    fn statement_digest_binds_seed_and_hm() {
        let codec = RiVeRCodec::new(RIVER_TOY);
        let mut rand = lcg(7);
        let q = codec.par.q();
        let h_m: Vec<Poly> = (0..codec.par.ell)
            .map(|_| (0..codec.par.d).map(|_| rand(q)).collect())
            .collect();

        let a = statement_digest(&codec, b"seed1", &h_m).unwrap();
        assert_eq!(a, statement_digest(&codec, b"seed1", &h_m).unwrap());
        assert_ne!(a, statement_digest(&codec, b"seed2", &h_m).unwrap());
        let mut h2 = h_m.clone();
        h2[0][0] = (h2[0][0] + 1) % q;
        assert_ne!(a, statement_digest(&codec, b"seed1", &h2).unwrap());
    }
}
