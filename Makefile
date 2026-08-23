# RiVeR -- recursive top-level Makefile.
#
# Nothing here knows what a subdirectory contains.  Every component owns its
# own Makefile and its own notion of `clean`, and this file only dispatches, so
# adding a component needs no edit here: drop in a Makefile with the standard
# targets and it joins recursive cleanup automatically.
#
# Standard targets a component Makefile should provide:
#
#   all clean distclean             every component
#   test test-all selftest kat      implementations (river-*)
#   vectors check-vectors           implementations (river-*)
#   bench bench-lanes bench-sizes   implementations with a Rust benchmark binary
#   table-check check               parameter-setting artifacts
#
# Discovered, not listed:
#   SUBDIRS   every immediate subdirectory holding a Makefile
#   IMPLDIRS  those matching `river-*`, i.e. the implementations
#
# Non-implementation components such as `parameters/` participate in
# `clean`/`distclean` through SUBDIRS but are not added to protocol tests.

SUBDIRS  := $(patsubst %/Makefile,%,$(wildcard */Makefile))
IMPLDIRS := $(patsubst %/Makefile,%,$(wildcard river-*/Makefile))
RUSTBENCHDIRS := $(patsubst %/src/bin/bench.rs,%,$(wildcard river-*/src/bin/bench.rs))
MANIFESTDIRS := $(patsubst %/manifest.py,%,$(wildcard river-*/manifest.py))

## Run one target across a list of subdirectories, or say nothing if empty.
# $(1) target, $(2) directory list
define recurse
@for d in $(2); do \
    echo "==> $$d: $(1)"; \
    $(MAKE) --no-print-directory -C $$d $(1) || exit 1; \
done
endef

.PHONY: help all test test-all selftest kat vectors check-vectors bench bench-lanes bench-sizes manifest \
        clean distclean

help:
	@echo "RiVeR"
	@echo
	@echo "  make test           run each implementation's test suite"
	@echo "  make test-all       the same, plus every published profile"
	@echo "  make selftest       per-module self-checks"
	@echo "  make kat            known-answer tests only (fast)"
	@echo "  make check-vectors  re-derive and diff the shipped test vectors"
	@echo "  make vectors        REGENERATE the test vectors (deliberate act)"
	@echo "  make bench          run the Rust implementation benchmarks"
	@echo "  make manifest       print the frozen wire-visible numeric manifest"
	@echo "  make bench-lanes    focused LANES ring/backend/codec benchmarks"
	@echo "  make bench-sizes    measured communication for every profile"
	@echo "  make clean          recurse; each component cleans itself"
	@echo "  make distclean      as clean, plus build directories"
	@echo
	@echo "components:      $(SUBDIRS)"
	@echo "implementations: $(IMPLDIRS)"
	@echo
	@echo "clean preserves shipped vectors, manifests, and parameter data/reports."

all: test

# ---- dispatched to the implementations -----------------------------------

test test-all selftest kat check-vectors:
	$(call recurse,$@,$(IMPLDIRS))

## Benchmarks are discovered separately from tests, because not every
## implementation has one and dispatching blindly would turn a missing
## benchmark into a missing-target error.  A component joins by shipping
## `src/bin/bench.rs`.
##
## Benchmarking is the Rust implementation's job alone: `river-py` is the
## golden reference and the vector generator, and a timing taken from it
## measures CPython rather than the protocol.
bench bench-lanes bench-sizes:
	$(call recurse,$@,$(RUSTBENCHDIRS))

## The wire-visible numeric manifest: pinned widths, Rice parameters and
## bounds, per field and profile.  Discovered the same way: a component
## joins by shipping `manifest.py`.
manifest:
	$(call recurse,$@,$(MANIFESTDIRS))

## Regenerating replaces the shipped reference.  Any change to the samplers,
## the codec, or a Fiat-Shamir input moves every byte, so a diff here is
## expected after such work -- and worth investigating after anything else.
vectors:
	@echo "This replaces the shipped reference vectors."
	@echo "Use 'make check-vectors' to verify without rewriting."
	$(call recurse,vectors,$(IMPLDIRS))

# ---- cleaning ------------------------------------------------------------
#
# Recursive: each component removes what it created.  Only root-level stray
# files are handled here, because only they belong to no component.

clean distclean:
	$(call recurse,$@,$(SUBDIRS))
	@rm -rf __pycache__ .pytest_cache
	@$(RM) *~ *.swp *.swo *.tmp gmon.out
	@echo "root: $@ complete"
