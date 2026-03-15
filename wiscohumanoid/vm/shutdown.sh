
echo "Attempting to shut down VM..."
RESPONSE=$(curl https://99t3dffvlc.execute-api.us-east-1.amazonaws.com/default/stop-gpu-vm1)

if [[ "$RESPONSE" == *"stopping"* ]]; then
  echo -e "\nSuccess!"
else
  echo -e "\n\e[1;31mERROR: VM shutdown FAILED for unknown reason, please message #alohamini ASAP!\e[0m"
fi