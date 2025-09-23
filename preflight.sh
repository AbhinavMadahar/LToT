REGION=us-east-1

# Latest DLAMI for Ubuntu 22.04 GPU/PyTorch via SSM (works with G6e/L40S)
AMI_ID=$(aws ssm get-parameter --region $REGION \
  --name /aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.7-ubuntu-22.04/latest/ami-id \
  --query "Parameter.Value" --output text)

# Default VPC → a default subnet → minimal security group
VPC_ID=$(aws ec2 describe-vpcs --region $REGION --filters Name=isDefault,Values=true \
  --query "Vpcs[0].VpcId" --output text)
SUBNET_ID=$(aws ec2 describe-subnets --region $REGION \
  --filters Name=vpc-id,Values=$VPC_ID Name=default-for-az,Values=true \
  --query "Subnets[0].SubnetId" --output text)
SG_ID=$(aws ec2 create-security-group --region $REGION --group-name ltot-sg \
  --description "LToT cluster" --vpc-id $VPC_ID --query "GroupId" --output text 2>/dev/null \
  || aws ec2 describe-security-groups --region $REGION \
       --filters Name=group-name,Values=ltot-sg Name=vpc-id,Values=$VPC_ID \
       --query "SecurityGroups[0].GroupId" --output text)
# (Optional) SSH from your IP only:
MYIP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG_ID \
  --protocol tcp --port 22 --cidr ${MYIP}/32 >/dev/null 2>&1 || true

echo "AMI_ID=$AMI_ID  SUBNET_ID=$SUBNET_ID  SG_ID=$SG_ID"
