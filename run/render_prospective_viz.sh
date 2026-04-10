#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

Rscript -e "rmarkdown::run('src/prospective_joint_twostage_viz.Rmd', shiny_args = list(launch.browser = TRUE))"
