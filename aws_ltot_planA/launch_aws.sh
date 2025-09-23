#!/usr/bin/env bash
set -euo pipefail

# ----------------- EDIT THESE -----------------
REGION="us-east-1"
AMI_ID="ami-xxxxxxxxxxxxxxxxx"          # Deep Learning AMI (Ubuntu 22.04) with CUDA
SUBNET_ID="subnet-xxxxxxxx"             # Private or public subnet in your VPC
SG_ID="sg-xxxxxxxx"                     # Security group (SSH from your IP; vLLM ports internal only)
IAM_INSTANCE_PROFILE="ltot-ec2-role"    # Instance profile name with S3 access
KEY_NAME="your-keypair"
S3_BUCKET="s3://ltot-yourbucket"        # e.g., s3://ltot-iclr
RUN_ID="run-$(date +%Y%m%d-%H%M%S)"     # change if desired
# Optional: export HF_TOKEN before running if you need to pull gated models.
# ---------------------------------------------

USER_DATA_70B="user-data-70b.yaml"
USER_DATA_8B="user-data-8b.yaml"
USER_DATA_MIXTRAL="user-data-mixtral-awq.yaml"

echo "[+] Launching 70B on-demand (fuse)"
aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type g6e.12xlarge \
  --count 1 \
  --key-name "$KEY_NAME" \
  --subnet-id "$SUBNET_ID" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile Name="$IAM_INSTANCE_PROFILE" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ltot-70b-fuse-ondemand},{Key=RunId,Value=$RUN_ID}]" \
  --user-data "file://$USER_DATA_70B"

echo "[+] Launching 70B Spot replicas (5 nodes)"
aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type g6e.12xlarge \
  --count 5 \
  --instance-market-options "MarketType=spot,SpotOptions={InstanceInterruptionBehavior=terminate}" \
  --key-name "$KEY_NAME" \
  --subnet-id "$SUBNET_ID" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile Name="$IAM_INSTANCE_PROFILE" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ltot-70b-spot},{Key=RunId,Value=$RUN_ID}]" \
  --user-data "file://$USER_DATA_70B"

echo "[+] Launching 8B servers (4 x single-GPU Spot g6e.xlarge)"
aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type g6e.xlarge \
  --count 4 \
  --instance-market-options "MarketType=spot,SpotOptions={InstanceInterruptionBehavior=terminate}" \
  --key-name "$KEY_NAME" \
  --subnet-id "$SUBNET_ID" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile Name="$IAM_INSTANCE_PROFILE" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ltot-8b-spot},{Key=RunId,Value=$RUN_ID}]" \
  --user-data "file://$USER_DATA_8B"

echo "[+] Launching Mixtral servers (4 x single-GPU Spot g6e.xlarge; AWQ)"
aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type g6e.xlarge \
  --count 4 \
  --instance-market-options "MarketType=spot,SpotOptions={InstanceInterruptionBehavior=terminate}" \
  --key-name "$KEY_NAME" \
  --subnet-id "$SUBNET_ID" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile Name="$IAM_INSTANCE_PROFILE" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ltot-mixtral-awq-spot},{Key=RunId,Value=$RUN_ID}]" \
  --user-data "file://$USER_DATA_MIXTRAL"

cat <<EOF

[INFO] Instances requested.
Next:
  1) Wait until all are "running" and checks passed.
  2) Collect their *private* IPs and fill endpoints.yaml:
     - 70B replicas: http://<ip>:8000
     - 8B servers:  http://<ip>:8100
     - Mixtral:     http://<ip>:8200
  3) python health_check.py endpoints.yaml

If you prefer *FP16 Mixtral* on 2 GPUs instead of AWQ:
  - Launch 2 x g6e.4xlarge Spot with a user-data that sets: --tensor-parallel-size 2 and remove --quantization.
EOF
