# Parameter-setting artifact.
#
# Generated tables and reports are shipped outputs and are deliberately
# preserved by `clean`. The cleanup target removes only caches, Sage preparse
# products outside scripts/, logs, and editor temporaries.

PYTHON ?= python3
SAGE ?= sage

define run_sage
$(SAGE) -c "import runpy, sys; sys.argv = ['$(1)']; runpy.run_path('$(1)', run_name='__main__')"
endef

.PHONY: all table-check check product-selftest product-check product-regenerate clean distclean

all: table-check

# Fast deterministic checks and table regeneration. This does not rerun the
# lattice estimators or the expanded finite-grid minimality diagnostic.
table-check:
	$(PYTHON) scripts/reproduce_final_table.py --check
	$(call run_sage,scripts/river_oom_math_checks.sage.py)
	$(call run_sage,scripts/validate_product_tau_inputs.py)
	$(PYTHON) scripts/make_all_parameters_table.py

# Full six-step reproduction, including the bundled estimators and expanded
# finite-grid diagnostic. See README.md for the diagnostic's current status.
check:
	bash scripts/run_all_checks.sh

# The deterministic aggregate replay is deliberately separate from `check`:
# its defaults run five one-million-trial experiments.
product-selftest:
	$(SAGE) -c "import runpy, sys; sys.argv = ['scripts/estimate_public_restart_bounds.sage.py', '--self-test-only']; runpy.run_path(sys.argv[0], run_name='__main__')"

product-check:
	$(SAGE) -c "import runpy, sys; sys.argv = ['scripts/estimate_public_restart_bounds.sage.py', '--check']; runpy.run_path(sys.argv[0], run_name='__main__')"

product-regenerate:
	$(call run_sage,scripts/estimate_public_restart_bounds.sage.py)

clean:
	find . -type d \( -name __pycache__ -o -name .sage \) -prune -exec rm -rf -- {} +
	find optional_challenge_invertibility external -type f -name '*.sage.py' -delete
	find . -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.sage.pyc' \
		-o -name '*.sobj' -o -name '*.spyx.c' -o -name 'sage_log*' \
		-o -name '*~' -o -name '*.swp' -o -name '*.swo' -o -name '*.tmp' \) -delete
	@echo "parameters: clean (data/ and report/ preserved)"

distclean: clean
