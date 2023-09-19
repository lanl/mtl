
# Only example training data are provided here
ntrain = 1
nvalid = 1
n1 = 360
n2 = 256
prefix = 'result2'

#===============================================================================
# Training
opts = "--n1=#{n1} --n2=#{n2} --ntrain=#{ntrain} --nvalid=#{nvalid} \
	--batch_train=8 --batch_valid=8 --lr=0.5e-4 --gpus_per_node=1 \
	--dir_data_train=dataset2/data_train \
	--dir_data_valid=dataset2/data_valid \
	--dir_target_train=dataset2/target_train \
	--dir_target_valid=dataset2/target_valid "

system "python ../src/main2_infer.py #{opts} --dir_output=#{prefix}_infer "
system "python ../src/main2_refine.py #{opts} --dir_output=#{prefix}_refine "

# ==============================================================================
# Validation
opts = "--n1=#{n1} --n2=#{n2} --nvalid=#{nvalid} \
	--batch_train=8 --batch_valid=8 --lr=0.5e-4 --gpus_per_node=1 \
	--dir_data_valid=dataset2/data_valid \
	--dir_target_valid=dataset2/target_valid "

system "python ../src/main2_infer.py #{opts} --check=#{prefix}_infer/trained.model --dir_output=#{prefix}_infer "
system "python ../src/main2_refine.py #{opts} --check=#{prefix}_refine/trained.model --dir_output=#{prefix}_refine "
