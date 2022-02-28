mkdir -p kernels/$1
cp kernel_import_template.ipynb kernels/$1/$1.ipynb
cp kernel-metadata_template.json kernels/$1/kernel-metadata.json
sed -i "s/<placeholder>/$1/g" kernels/$1/kernel-metadata.json

echo "New kernel prepared in kernels/$1."