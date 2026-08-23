PYTHON ?= python3

TESTS = test_kat test_params test_manifest test_ring test_sample test_codec \
        test_lanes_ring test_lanes_manifest \
        test_exact test_oom test_river test_e2e test_review \
        test_lanes

MODULES = params manifest ring sample dgs codec exact oom river \
          lanes_ring lanes_params lanes_commit lanes_mp lanes_proof \
          lanes_backend lanes_manifest

.PHONY: all test test-all selftest kat vectors check-vectors \
        manifest manifest-regen clean distclean \
        lanes-manifest lanes-manifest-regen lanes-manifest-check

all: test

## run every unit / end-to-end test file
test:
	@for t in $(TESTS); do \
	    echo "== $$t"; \
	    $(PYTHON) $$t.py || exit 1; \
	done

## as `test`, plus the slow all-profile sweep (N=16..256)
test-all:
	@RIVER_SLOW_TESTS=1 $(MAKE) test

## primitive known-answer tests only; the fastest useful check
kat:
	$(PYTHON) test_kat.py

## run the per-module __main__ self-tests
selftest:
	@for m in $(MODULES); do \
	    echo "== $$m"; \
	    $(PYTHON) $$m.py || exit 1; \
	done

## Print the frozen wire-visible numeric manifest (widths, Rice parameters,
## bounds, both layouts, framing).  `test_manifest.py` compares
## `manifest.json` byte for byte; this is how to read it.
manifest:
	$(PYTHON) manifest.py

## REWRITE `manifest.json`.  Deliberate, like `make vectors`: it moves the
## frozen wire description the Rust port is built against.
## Print the frozen LANES parameter manifest (experimental; see
## `lanes_manifest.py`).  `test_lanes_manifest.py` checks the code against
## it, which is why regenerating it is deliberate.
lanes-manifest:
	$(PYTHON) lanes_manifest.py

## REWRITE `lanes_manifest.json`.  Deliberate, like `make vectors`.
lanes-manifest-regen:
	$(PYTHON) lanes_manifest.py --write

## Check the frozen LANES manifest against the live modules.
lanes-manifest-check:
	$(PYTHON) lanes_manifest.py --check

manifest-regen:
	@echo "This replaces the frozen wire manifest."
	$(PYTHON) manifest.py --write

## regenerate the shipped test vectors
vectors:
	$(PYTHON) vectors.py --out vectors.json

## re-derive everything and compare against the shipped vectors
check-vectors:
	$(PYTHON) vectors.py --verify vectors.json

## Remove this directory's build artifacts and temporary files.
##
## `vectors.json`, `manifest.json` and `lanes_manifest.json` are
## deliberately NOT removed: all three are frozen deliverables that the
## test suite checks the code *against*, and each is regenerated only by
## its own deliberate target (`make vectors`, `make manifest-regen`,
## `make lanes-manifest-regen`).  Keep this set in step with the
## repository .gitignore.
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache .ty_cache
	$(RM) *.pyc *.pyo *.pyd *.so
	$(RM) *~ *.swp *.swo *.tmp
	@echo "river-py: clean (vectors.json, manifest.json, lanes_manifest.json preserved)"

## Nothing further to remove here; present so the recursive target exists.
distclean: clean
