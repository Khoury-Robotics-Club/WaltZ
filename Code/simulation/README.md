## TO RUN

### RUN Soft actor critic
* For multiporcessing script use the below command to execute the code
* nohup python3 sac_multi.py > output.log 2>&1 && echo "DONE" > ./done &
* For a single threaded code run the below command
* nohup python3 sac.py > output.log 2>&1 && echo "DONE" > ./done &
