# IAM
IAM_ROLE_ARN ='arn:aws:iam::385348158986:instance-profile/LabInstanceProfile'

# S3
S3_NAME ='matrixs3operfkemngotnspoff'
LOCAL_DIR_TO_COPY='script'

# Security Group
GROUP_NAME = 'matrixInstanceSecurityGroup'

# EC2
INSTANCE_LIST_NAME=['worker1', 'worker2', 'worker3', 'worker4', 'worker5', 'worker6', 'worker7', 'master']
IMAGE_ID='ami-08e637cea2f053dfa'
INSTANCE_TYPE_WORKER='t2.micro'
INSTANCE_TYPE_MASTER='t3.large'
KEY_NAME='vockey'
USER_DATA_WORKER='''#!/bin/bash
sudo yum update -y
sudo yum -y install zip unzip
curl "https://s3.amazonaws.com/aws-cli/awscli-bundle.zip" -o "awscli-bundle.zip"
unzip awscli-bundle.zip
sudo ./awscli-bundle/install -i /usr/local/aws -b /usr/local/bin/aws
sudo yum -y install python-pip
pip install boto3 numpy
aws s3 cp s3://matrixs3operfkemngotnspoff/workerCode.py home/ec2-user/code/workerCode.py
'''
USER_DATA_MASTER='''#!/bin/bash
sudo yum update -y
sudo yum -y install zip unzip
curl "https://s3.amazonaws.com/aws-cli/awscli-bundle.zip" -o "awscli-bundle.zip"
unzip awscli-bundle.zip
sudo ./awscli-bundle/install -i /usr/local/aws -b /usr/local/bin/aws
sudo yum -y install python-pip
pip install boto3 streamlit
pip install --upgrade jinja2
aws s3 cp s3://matrixs3operfkemngotnspoff/masterCode.py home/ec2-user/code/masterCode.py
aws s3 cp s3://matrixs3operfkemngotnspoff/appStreamLit.py home/ec2-user/code/appStreamLit.py
'''

#SQS
QUEUE_LIST_NAME=['matrixQueueMtoW', 'matrixQueueWtoM']

# Other
LABSUSER_PATH='/home/anthony/Desktop/labsuser.pem'

