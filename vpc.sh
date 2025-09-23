# 0) Region
REGION=us-east-1

# 1) Make sure a default VPC exists (create it if missing)
VPC_ID=$(aws ec2 describe-vpcs --region $REGION \
  --filters Name=isDefault,Values=true \
  --query "Vpcs[0].VpcId" --output text)

if [ "$VPC_ID" = "None" ] || [ -z "$VPC_ID" ]; then
  echo "[i] No default VPC; creating one..."
  VPC_ID=$(aws ec2 create-default-vpc --region $REGION \
    --query "Vpc.VpcId" --output text)
fi
echo "[+] VPC_ID: $VPC_ID"

# 2) Pick a default subnet (these auto-assign public IPs and have internet egress)
SUBNET_ID=$(aws ec2 describe-subnets --region $REGION \
  --filters Name=vpc-id,Values=$VPC_ID Name=default-for-az,Values=true \
  --query "Subnets[0].SubnetId" --output text)
echo "[+] SUBNET_ID: $SUBNET_ID"

# (Optional) sanity check it's public/auto-assigning:
aws ec2 describe-subnets --region $REGION --subnet-ids $SUBNET_ID \
  --query "Subnets[0].MapPublicIpOnLaunch"

# 3) Create (or reuse) a minimal security group
SG_NAME=ltot-sg
SG_ID=$(aws ec2 create-security-group --region $REGION \
  --group-name $SG_NAME --description "LToT cluster" \
  --vpc-id $VPC_ID --query "GroupId" --output text 2>/dev/null \
  || aws ec2 describe-security-groups --region $REGION \
       --filters Name=group-name,Values=$SG_NAME Name=vpc-id,Values=$VPC_ID \
       --query "SecurityGroups[0].GroupId" --output text)
echo "[+] SG_ID: $SG_ID"

# 4) (Optional) allow SSH only from your current IP; otherwise skip this
MYIP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --region $REGION \
  --group-id $SG_ID --protocol tcp --port 22 --cidr ${MYIP}/32 \
  >/dev/null 2>&1 || true
echo "[+] SSH allowed from ${MYIP}/32 (optional). No other inbound ports are open."
