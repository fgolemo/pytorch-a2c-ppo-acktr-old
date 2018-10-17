good:

python3 pytorch_a2c_ppo_acktr/enjoy.py --env-name Duckietown-loop_obstacles-v0 --custom-gym gym_duckietown --model ./trained_models/ppo/Duckietown-loop_obstacles-v0-181015135857-best.pt --num-stack 4 --seed 1 --scale-img --duckietown

(no color, ppo, discrete (5))


kinda:

python3 pytorch_a2c_ppo_acktr/enjoy.py --env-name Duckietown-loop_obstacles-v0 --custom-gym gym_duckietown --model ./trained_models/ppo/Duckietown-loop_obstacles-v0-181015135658-best.pt --num-stack 1 --seed 1 --scale-img --duckietown

(ppo, color, discrete (5))


