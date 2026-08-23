#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

run_sage() {
    local script=$1
    sage -c "import runpy, sys; sys.argv = ['${script}']; runpy.run_path('${script}', run_name='__main__')"
}

python3 scripts/reproduce_final_table.py --check
run_sage scripts/river_oom_math_checks.sage.py
run_sage scripts/validate_product_tau_inputs.py
run_sage scripts/run_final_oom_estimators.sage.py
minimality_status=0
run_sage scripts/verify_oom_search_minimality.sage.py || minimality_status=$?
python3 scripts/make_all_parameters_table.py
exit "$minimality_status"
