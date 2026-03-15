#!/bin/bash

REMINDER="\n\e[1;31m===============================================================================\n*** ALOHAMINI AWS LOGIN - MANUALLY TURN OFF USING SHUTDOWN.SH WHEN COMPLETE ***\n===============================================================================\e[0m"

trap 'echo -e "\n${REMINDER}"' EXIT

echo -e "${REMINDER}\n"
echo "Starting AWS instance..."
curl https://f52k3dl13j.execute-api.us-east-1.amazonaws.com/default/start-gpu-vm1

echo -e "\nAttempting SSH connection, this can take up to ~2 mins depending on cloud availability..."
ssh -i ~/.ssh/gpu-vm1-key.pem ubuntu@44.193.239.214