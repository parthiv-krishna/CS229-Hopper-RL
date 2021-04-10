models=([0]="ddqn" [1]="ddpg" [2]="ppo", [3]="rand")

env=$1
model=$2
steps=$3
baseport=$4
workers=16
trainprop=1

open_terminal()
{
	script=$1
	if [[ "$OSTYPE" == "darwin"* ]]; then # Running on mac
		osascript <<END 
		tell app "Terminal" to do script "cd \"`pwd`\"; $script; exit"
END
	elif [[ "$OSTYPE" == "linux-gnu" ]]; then # Running on linux
		xterm -e $script $2 # Add -hold argument after xterm to debug
	fi
}

run()
{
	steps=$1
	env=$2
	agent=$3
	numWorkers=$4

	echo "Workers: $numWorkers, Steps: $steps, Agent: $agent"

	ports=()
	for j in `seq 0 $numWorkers` 
	do
		port=$((8000+$baseport+$j))
		ports+=($port)
	done
	port_string=$( IFS=$' '; echo "${ports[*]}" )

	for j in `seq $numWorkers -1 1` 
	do
		open_terminal "julia julia/train_agent.jl $env $agent --nsteps $steps --tcp_rank $j --tcp_ports $port_string --train_prop $trainprop" &
		sleep 0.4
	done
	sleep 30
	julia julia/train_agent.jl $env $agent --nsteps $steps --tcp_rank 0 --tcp_ports $port_string --train_prop $trainprop
}

run $steps $env $model $workers