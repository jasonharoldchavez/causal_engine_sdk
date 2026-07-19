import subprocess

for t in [0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0]:

    with open("real_data_tests/base_asie_template.py") as f:
        code=f.read().replace("THRESH_VALUE",str(t))

    with open("real_data_tests/run_asie.py","w") as f:
        f.write(code)

    out=subprocess.check_output(
        ["python3","real_data_tests/run_asie.py"]
    ).decode()

    print("\nTHRESHOLD:",t)
    print(out)
