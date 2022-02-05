.PHONY: submit_latest show_submissions download_dataset

comp_name := happy-whale-and-dolphin
latest_subm_file := $(shell ls -t submissions/ | head -n1)
base_dir := $(shell pwd)
file_stem = $(word 2, $(subst ., ,$(latest_subm_file)))

submit_latest: 
	kaggle competitions submit -f $(base_dir)/submissions/$(latest_subm_file) -c $(comp_name) -m $(file_stem)

show_submissions:
	kaggle competitions submissions $(comp_name)

download_dataset:
	kaggle competitions download -c $(comp_name) -p data/

test:
	echo $(bla_2)
	