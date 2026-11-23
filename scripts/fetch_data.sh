#!/usr/bin/env bash
# Downloads the third-party inputs (not redistributed here) into external/ in the layout the scripts expect.
# Needs git, curl and unzip. About 1.3 GB, mostly the Chiang & Lee repository.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external && cd external
clone() { [ -d "$2" ] || git clone -q --depth 1 "https://github.com/$1" "$2"; }
clone nlpyang/geval geval                                        # G-Eval: results/gpt4_*_detailed.json
clone krystalan/chatgpt_as_nlg_evaluator wang                    # Wang et al.: data/summeval.json
clone DAMO-NLP-SG/LLM_summeval shen_repo                         # Shen et al.
clone d223302/A-Closer-Look-To-LLM-Evaluation chiang             # Chiang & Lee: results/{summeval,topical_chat}
# Shen et al.: the script expects shen/summeval.json and shen/<model>/<M8..M23>/<aspect>_<prompt>.txt
mkdir -p shen && cp shen_repo/summeval.json shen/
for m in gpt-4-0314 gpt-3.5-turbo-0301; do
  mkdir -p "shen/$m"
  for d in shen_repo/eval_model_generations/$m/eval_M*_generations; do
    M=$(basename "$d"); M=${M#eval_}; M=${M%_generations}
    rm -rf "shen/$m/$M"; cp -r "$d" "shen/$m/$M"
  done
done
# Wine Quality (Cortez et al., UCI, CC BY 4.0), for Appendix B only
if [ ! -f wine/winequality-red.csv ]; then
  mkdir -p wine && curl -sSL -o wine/wine.zip https://archive.ics.uci.edu/static/public/186/wine+quality.zip \
    && unzip -o -q wine/wine.zip -d wine || echo "WARNING: wine download failed; Appendix B wine rows will be skipped"
fi
echo "inputs ready in external/"
