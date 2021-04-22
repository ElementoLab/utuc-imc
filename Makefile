.DEFAULT_GOAL := all

NAME=$(shell basename `pwd`)
SAMPLES=$(shell ls data)
PANEL=metadata/panel_markers.UTUC.csv
MODEL=_models/$(NAME)/$(NAME).ilp

help:  ## Display help and quit
	@echo Makefile for the $(NAME) project/package.
	@echo Available commands:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m\
		%s\n", $$1, $$2}'

all: install clean test

requirements:  ## Install Python requirements
	pip install -r requirements.txt

transfer:  ## Transfer data from wcm.box.com to local environment
	imctransfer -q 2020  # Query for files produced in 2020 only

prepare:  ##  Run first step of convertion of MCD to various files
	@echo "Running prepare step for samples: $(SAMPLES)"
	mkdir -p processed
	for SAMPLE in $(SAMPLES); do \
	python -u src/_prepare_mcd.py \
			--n-crops 1 \
			data/$${SAMPLE}/$${SAMPLE}.mcd \
			$(PANEL) \
			-o processed/$${SAMPLE}; \
	done

process_local:  ## Run IMC pipeline locally
	@echo $(SAMPLES)
	for SAMPLE in $(SAMPLES); do \
		python -u -m imcpipeline.pipeline \
		--ilastik-model $(MODEL) \
		--csv-pannel $(PANEL) \
		--container docker \
		-i data/$${SAMPLE} \
		-o processed/$${SAMPLE} \
		-s predict,segment; \
	done

process_scu:  ## Run IMC pipeline on SCU
	for SAMPLE in $(SAMPLES); do
		# python -u ~/projects/imcpipeline/imcpipeline/pipeline.py \
		python -u -m imcpipeline.pipeline \
		--ilastik-model $(MODEL) \
		--csv-pannel $(PANEL) \
		--cellprofiler-exec \
			"source ~/.miniconda2/bin/activate && conda activate cellprofiler && cellprofiler" \
		-i data/$${SAMPLE} \
		-o processed/$${SAMPLE} \
		-s predict,segment
	done

run:
	imcrunner \
		--divvy slurm \
		metadata/samples.initial.csv \
			--ilastik-model $(MODEL) \
			--csv-pannel $(PANEL) \
			--cellprofiler-exec \
				"source ~/.miniconda2/bin/activate && conda activate cellprofiler && cellprofiler"

run_locally:
	imcrunner \
		--divvy local \
		metadata/samples.initial.csv \
			--ilastik-model $(MODEL) \
			--csv-pannel $(PANEL) \
			--container docker


checkfailure:  ## Check whether any samples failed during preprocessing
	grep -H "Killed" submission/*.log && \
	grep -H "Error" submission/*.log && \
	grep -H "CANCELLED" submission/*.log && \
	grep -H "exceeded" submission/*.log

fail: checkfailure  ## Check whether any samples failed during preprocessing

checksuccess:  ## Check which samples succeded during preprocessing
	ls -hl processed/*/cpout/cell.csv

succ: checksuccess  ## Check which samples succeded during preprocessing


rename_outputs:  ## Rename outputs from CellProfiler output to values expected by `imc`
	find processed \
		-name "*_ilastik_s2_Probabilities.tiff" \
		-exec rename -f "s/_ilastik_s2_Probabilities/_Probabilities/g" \
		{} \;
	find processed \
		-name "*_ilastik_s2_Probabilities_mask.tiff" \
		-exec rename -f "s/_ilastik_s2_Probabilities_mask/_full_mask/g" \
		{} \;
	find processed \
		-name "*_ilastik_s2_Probabilities_NucMask.tiff" \
		-exec rename -f "s/_ilastik_s2_Probabilities_NucMask/_full_nucmask/g" \
		{} \;

rename_outputs_back:  ## Rename outputs from values expected by `imc` to CellProfiler
	find processed \
		-name "*_Probabilities.tiff" \
		-exec rename -f "s/_Probabilities/_ilastik_s2_Probabilities/g" \
		{} \;
	find processed \
		-name "*_full_mask.tiff" \
		-exec rename -f "s/_full_mask/_ilastik_s2_Probabilities_mask/g" \
		{} \;
	find processed \
		-name "*_full_nucmask.tiff" \
		-exec rename -f "s/_full_nucmask/_ilastik_s2_Probabilities_NucMask/g" \
		{} \;

merge_runs:  ## Merge images from the same acquisition that were in multiple MCD files
	python -u src/_merge_runs.py

analysis:
	python -u src/illustration.py
	python -u src/analysis.py

backup_time:
	echo "Last backup: " `date` >> _backup_time
	chmod 700 _backup_time

_sync:
	rsync --copy-links --progress -r \
	. afr4001@pascal.med.cornell.edu:projects/$(NAME)

sync: _sync backup_time ## [dev] Sync data/code to SCU server (should be done only when processing files from MCD files)


.PHONY : move_models_out move_models_in clean_build clean_dist clean_eggs \
clean _install install clean_docs docs run run_locally \
checkfailure fail checksuccess succ backup_time _sync sync
