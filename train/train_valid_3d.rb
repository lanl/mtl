
# Only example training data are provided here
ntrain = 1
nvalid = 1
#n1 = 360
#n2 = 128
#n3 = 128
n1 = 64
n2 = 64
n3 = 64
prefix = 'result3'

#===============================================================================
# Training
opts = "--n1=#{n1} --n2=#{n2} --n3=#{n3} --ntrain=#{ntrain} --nvalid=#{nvalid} \
	--batch_train=8 --batch_valid=8 --lr=0.5e-4 --gpus_per_node=1 \
	--dir_data_train=dataset3/data_train \
	--dir_data_valid=dataset3/data_valid \
	--dir_target_train=dataset3/target_train \
	--dir_target_valid=dataset3/target_valid "

system "python ../src/main3_infer.py #{opts} --dir_output=#{prefix}_infer "
system "python ../src/main3_refine.py #{opts} --dir_output=#{prefix}_refine "

# ==============================================================================
# Validation
opts = "--n1=#{n1} --n2=#{n2} --n3=#{n3} --nvalid=#{nvalid} \
	--batch_train=8 --batch_valid=8 --lr=0.5e-4 --gpus_per_node=1 \
	--dir_data_valid=dataset3/data_valid \
	--dir_target_valid=dataset3/target_valid "

system "python ../src/main3_infer.py #{opts} --check=#{prefix}_infer/trained.model --dir_output=#{prefix}_infer "
system "python ../src/main3_refine.py #{opts} --check=#{prefix}_refine/trained.model --dir_output=#{prefix}_refine "
