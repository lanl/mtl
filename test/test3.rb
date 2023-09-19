
model_infer = "../train/result3_infer/trained.model"
model_refine = "../train/result3_refine/trained.model"

nn1 = 128
nn2 = 256
nn3 = 128
file = "opunake3"
clip = 1
clip0 = 0.25e5

# ==============================================================================
# Inference
system "python ../src/main3_infer.py --gpus_per_node=1 --model=#{model_infer} \
	--n1=#{nn1} --n2=#{nn2} --n3=#{nn3} --input=#{file} --output=#{file}.iter0 "

# ==============================================================================
# Refinement
niter = 3
for i in 1..niter
    system "python ../src/main3_refine.py --gpus_per_node=1 --model=#{model_refine} \
    	--n1=#{nn1} --n2=#{nn2} --n3=#{nn3} --input=#{file}.iter#{i - 1} --output=#{file}.iter#{i} "
end

# ==============================================================================
# Plot
opts = " -lmtick=4 -label1='Z (Grid Number)' -label2='Y (Grid Number)' -label3='X (Grid Number)' -mtick1=4 -mtick2=9 -tick1d=50 -tick2d=50 -tick3d=50 -size1=2.5 -size2=#{nn2*1.0/nn1*2.5} -size3=#{nn3*1.0/nn1*2.5} -n1=#{nn1} -n2=#{nn2} "

system "x_showslice #{opts} -in=#{file} -color=binary -out=#{file}.raw.pdf -clip=#{clip0} &"
system "x_showslice #{opts} -background=#{file}.iter#{niter}.dhr -in=#{file}.iter#{niter}.rgt -color=jet \
        -alphas=0:0.5,1:0.5 -backcolor=binary -out=#{file}.rgt.pdf -backclip=#{clip} -legend=y -unit='RGT' -cmin=0 -cmax=1 &"
system "x_showslice #{opts} -background=#{file}.iter#{niter}.dhr -cscale=180 -in=#{file}.iter#{niter}.fdip \
        -backcolor=binary  -color=jet -out=#{file}.dip.pdf -backclip=#{clip} -legend=y -unit='Fault Dip (degree)' \
        -alphas=60:0,60.1:1 -cmin=60 -cmax=120 -ld=10 -lmtick=9  &"
system "x_showslice #{opts} -background=#{file}.iter#{niter}.dhr -cscale=180 -in=#{file}.iter#{niter}.fstrike \
        -backcolor=binary  -color=jet -out=#{file}.strike.pdf -backclip=#{clip} -legend=y -unit='Fault Strike (degree)' \
        -alphas=0:0,0.1:1 -cmin=0 -cmax=180 -ld=10 -lmtick=9  &"
