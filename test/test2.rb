
model_infer = "../train/result2_infer/trained.model"
model_refine = "../train/result2_refine/trained.model"

nn1 = 256
nn2 = 1024
file = "opunake2"
clip = 1
clip0 = 0.5

# ==============================================================================
# Inference
system "python ../src/main2_infer.py --gpus_per_node=1 --model=#{model_infer} \
	--n1=#{nn1} --n2=#{nn2} --input=#{file} --output=#{file}.iter0 "

# ==============================================================================
# Refinement
niter = 3
for i in 1..niter
    system "python ../src/main2_refine.py --gpus_per_node=1 --model=#{model_refine} \
    	--n1=#{nn1} --n2=#{nn2} --input=#{file}.iter#{i - 1} --output=#{file}.iter#{i} "
end

# ==============================================================================
# Plot
opts = " -lheight=2.3 -lmtick=4 -label1='Z (Grid Number)' -label2='X (Grid Number)' \
	-mtick1=4 -mtick2=9 -tick1d=50 -tick2d=100 -size1=2.5 -size2=#{nn2*1.0/nn1*2.5} -n1=#{nn1} "

system "x_showmatrix #{opts} -in=#{file}.resamp -color=binary -out=#{file}.raw.pdf -clip=#{clip0} &"
system "x_showmatrix #{opts} -background=#{file}.iter#{niter}.dhr -in=#{file}.iter#{niter}.rgt -color=jet \
	-alphas=0:0.5,1:0.5 -backcolor=binary -out=#{file}.rgt.pdf -backclip=#{clip} -legend=y -unit='RGT' -cmin=0 -cmax=1 &"
system "x_showmatrix #{opts} -background=#{file}.iter#{niter}.dhr -cscale=180 -in=#{file}.iter#{niter}.fdip \
	-backcolor=binary  -color=jet -out=#{file}.dip.pdf -backclip=#{clip} -legend=y -unit='Fault Dip (degree)' \
	-alphas=60:0,60.1:1 -cmin=60 -cmax=120 -ld=10 -lmtick=9  &"
