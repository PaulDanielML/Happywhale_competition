.PHONY: submit_latest show_submissions download_dataset update_utils_dataset

comp_name := happy-whale-and-dolphin
latest_subm_file := $(shell ls -t submissions/ | head -n1)
base_dir := $(shell pwd)
file_stem = $(word 1, $(subst ., ,$(latest_subm_file)))

submit_latest: 
	kaggle competitions submit -f $(base_dir)/submissions/$(latest_subm_file) -c $(comp_name) -m $(file_stem)

show_submissions:
	kaggle competitions submissions $(comp_name)

download_dataset:
	kaggle competitions download -c $(comp_name) -p data/

update_utils_dataset:
	kaggle datasets version -p src/utils -m "update utils" 

new_kernel:
	mkdir -p kernels/$(name)
	cp kernel_import_template.ipynb kernels/$(name)/$(name).ipynb
	cp kernel-metadata_template.json kernels/$(name)/kernel-metadata.json
	sed -i "s/<placeholder>/$(name)/g" kernels/$(name)/kernel-metadata.json

	echo "New kernel prepared in kernels/$(name)."

push:
	kaggle kernels push -p kernels/$(name)

pull:
	kaggle kernels pull -p kernels/$(name) sharppa/$(name)

check:
	kaggle kernels status sharppa/$(name)
	
update-common-utils:
	pip install -I --no-dependencies git+http://github.com/PaulDanielML/common_utils.git