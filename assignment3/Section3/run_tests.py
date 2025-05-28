import subprocess
import itertools

# Define the three sets of parameter combinations
param_sets = [
    {
        'file': 'result1.txt',
        'b_values': [128],
        'p_values': [1024],
        'd_values': [2**5, 2**6, 2**7, 2**8, 2**9, 2**10],
    },
    {
        'file': 'result2.txt',
        'b_values': [1],
        'p_values': [2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16],
        'd_values': [128], # What is the decode length here? Does it matter?
    },
    {
        'file': 'result3.txt',
        'b_values': [2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10],
        'p_values': [128],
        'd_values': [128],
    }
]

for params in param_sets:
    with open(params['file'], 'w') as f:
        for b, p, d in itertools.product(params['b_values'], params['p_values'], params['d_values']):
            header = f"\nRunning with -b {b} -p {p} -d {d}\n"
            print(header.strip())
            f.write(header)

            cmd = ['python3', 'flashinfer_pipeline.py', '-b', str(b), '-p', str(p), '-d', str(d)]

            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                output = result.stdout
                relevant_lines = [line for line in output.splitlines() if line.strip().startswith('>>>')]
                for line in relevant_lines:
                    print(line)
                    f.write(line + '\n')

            except subprocess.CalledProcessError as e:
                error_msg = f"Command failed with return code {e.returncode}\n{e.stderr}\n"
                print(error_msg.strip())
                f.write(error_msg)
