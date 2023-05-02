import multiprocessing
import subprocess

# define a function to run your program
def run_program(args):
    print("args: " + str(args))
    subprocess.call(['python', 'ea.py', '--csv_output', args[0], "--accel",str(args[1])])

if __name__ == '__main__':
    # specify the number of parallel processes to run
    num_processes = 10
    
    # create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_processes)

    """Normal"""
    # create a list of arguments for each parallel process
    args = [('simulation_results/normal_output_forward_only'+ str(i),3) for i in range(10)]
    
    # map the function to the arguments and run in parallel
    pool.map(run_program, args)
    
    """slow"""
    # create a list of arguments for each parallel process
    args = [('simulation_results/slow_output_forward_only'+ str(i),0.3) for i in range(10)]
    
    # create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_processes)
    
    # map the function to the arguments and run in parallel
    pool.map(run_program, args)

    """fast"""
    # create a list of arguments for each parallel process
    args = [('simulation_results/fast_output_forward_only'+ str(i),9.81) for i in range(10)]
    
    # map the function to the arguments and run in parallel
    pool.map(run_program, args)

    """very fast"""
    # create a list of arguments for each parallel process
    args = [('simulation_results/very_fast_output_forward_only'+ str(i),9.81*5) for i in range(10)]
    
    # map the function to the arguments and run in parallel
    pool.map(run_program, args)

    # close the pool
    pool.close()

