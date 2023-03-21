import boto3
import shared
from botocore.exceptions import ClientError
import os


def initClient(service_name):
    """Create a client for the desired AWS service.

    Args:
        service_name (string): AWS service.

    Returns:
        client: Service Client
    """
    client = boto3.client(service_name)
    return client


def initResource(service_name):
    """Create a resource for the desired AWS service.

    Args:
        service_name (string): AWS service.

    Returns:
        resource: Service Resource
    """
    resource = boto3.resource(service_name)
    return resource


###################################################### S3 ##############################################################

def getBucketsList(s3_client):
    """Gives all S3 buckets present in the VPC.

    Args:
        s3_client (S3): S3 client.

    Returns:
        buckets_list (list): List of bucket names
    """
    response = s3_client.list_buckets()
    buckets_list = [bucket['Name'] for bucket in response['Buckets']]

    return buckets_list


def createBucket(s3_client, bucket_name, buckets_list):
    """Create the desired S3 bucket for the environment only if it is not already created

    Args:
        s3_client (S3): S3 client.
        bucket_name (string): Bucket name necessary to our application.
        buckets_list (list): List of bucket names
    """
    if bucket_name not in buckets_list:
        # Create the bucket
        s3_client.create_bucket(Bucket=bucket_name)
        print(f'Bucket {bucket_name} created')
    else:
        print(f'Bucket {bucket_name} already exists')


def fillBucket(s3_client, local_directory, bucket_name):
    """Upload scripts needed for the application from a local directory into an S3 bucket.

    Args:
        s3_client (S3): S3 client.
        local_directory (string): Path of the local directory where are store the scripts.
        bucket_name (string): Name of the S3 bucket where the script will be uploaded.
    """
    local_directory = local_directory

    # Loop through the files in the local directory
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            # Construct the full path of the file
            local_path = os.path.join(root, file)
            # Construct the key of the file in the bucket
            remote_path = os.path.relpath(local_path, local_directory)
            # Upload the file to the bucket
            s3_client.upload_file(local_path, bucket_name, remote_path)
            print(f'{local_path} copy to {bucket_name}')


##################################################### EC2 ##############################################################


class Instance:
    """A class representing am EC2 instance."""
    def __init__(self, id, value, publicIp='0.0.0.0', name=''):
        """Initialize a new EC2 instance.

        Args:
        id (int): The name of the student.
        name (string): The name of the EC2 instance.
        value (string): The value of the EC2 instance representing its state.
        publicIp (string): The public Ip address of the EC2
        """
        self.id = id
        self.name = name
        self.value = value
        self.publicIp = publicIp



def startInstance(ec2_client, instance_id):
    """Start an EC2 instance stopped.

    Args:
            ec2_client (EC2): EC2 client.
            instance_id (string): ID of the instance to start
    """

    # Do a dryrun first to verify permissions
    try:
        ec2_client.start_instances(InstanceIds=[instance_id], DryRun=True)
    except ClientError as e:
        if 'DryRunOperation' not in str(e):
            raise

    # Dry run succeeded, run start_instances without dryrun
    try:
        response = ec2_client.start_instances(InstanceIds=[instance_id], DryRun=False)
        print(response)
    except ClientError as e:
        print(e)


def createInstance(ec2_client, name, securityGroupId):
    """Create an EC2 instance with his specifications.

    Args:
        ec2_client (EC2): EC2 client.
        name (string): Desired name for the instance.
        securityGroupId (string): ID of the security group associate with the instance
    """

    if name == 'master':
        # Create a new EC2 instance
        response = ec2_client.run_instances(
            ImageId=shared.IMAGE_ID,
            InstanceType=shared.INSTANCE_TYPE_MASTER,
            MinCount=1,  # Minimum number of instances to launch
            MaxCount=1,  # Maximum number of instances to launch
            KeyName=shared.KEY_NAME,
            SecurityGroupIds=[securityGroupId],
            IamInstanceProfile={
                'Arn': shared.IAM_ROLE_ARN
            },
            UserData=shared.USER_DATA_MASTER
        )
    else:
        # Create a new EC2 instance
        response = ec2_client.run_instances(
            ImageId=shared.IMAGE_ID,
            InstanceType=shared.INSTANCE_TYPE_WORKER,
            MinCount=1,  # Minimum number of instances to launch
            MaxCount=1,  # Maximum number of instances to launch
            KeyName=shared.KEY_NAME,
            SecurityGroupIds=[securityGroupId],
            IamInstanceProfile={
                'Arn': shared.IAM_ROLE_ARN
            },
            UserData=shared.USER_DATA_WORKER
        )

    # Get the instance ID of the new instance
    instance_id = response['Instances'][0]['InstanceId']

    # Add a name to the instance
    ec2_client.create_tags(
        Resources=[instance_id],
        Tags=[{'Key': 'Name', 'Value': name}]
    )

    print(response)


def getAllInstance(ec2_client):
    """Returns all instances present in the VPC

    Args:
        ec2_client (EC2): EC2 client.

    Returns:
        instance_list (list[Instance]): A list of Instance object
    """
    instance_list = []

    Myec2 = ec2_client.describe_instances()

    # Init an array fill with all the information about the instance of the current env
    for group in Myec2['Reservations']:
        for instances in group['Instances']:
            try:
                i = Instance(instances['InstanceId'], instances['State']['Name'], publicIp=instances['PublicIpAddress'], name=instances['Tags'][0]['Value'])
            except:
                try:
                    i = Instance(instances['InstanceId'], instances['State']['Name'], name=instances['Tags'][0]['Value'])
                except ClientError as e:
                    i = Instance(instances['InstanceId'], instances['State']['Name'])
            instance_list.append(i)

    return instance_list


def initAllInstances(ec2_client, instance_list, expected_instance_list, security_group_id):
    """Checks if the desired instances are created, if they are, launches the instance otherwise creates them

    Args:
        ec2_client (EC2): EC2 client.
        instance_list (list[Instance]): A list of Instance object
        expected_instance_list (list[string]): A list of th expected instance name
        security_group_id (string): ID of the security group
    """
    temp_instance_list = []

    for instance in instance_list:
        if instance.name in expected_instance_list:
            if (instance.value == 'running' or instance.value == 'pending'):
                temp_instance_list.append(instance.name)
            if instance.value == 'stopped':
                temp_instance_list.append(instance.name)
                startInstance(ec2_client, instance.id)

    for expected_instance_name in expected_instance_list:
        if expected_instance_name not in temp_instance_list:
            createInstance(ec2_client, expected_instance_name, security_group_id)


################################################ Security Group ########################################################

def getAllSecurityGroup(ec2_client):
    """Returns all security group present in the VPC

    Args:
        ec2_client (EC2): EC2 client.

    Returns:
        security_group_list (list[obj]): A list of security group
    """
    # Call the describe_security_groups method to retrieve a list of all security groups
    security_group_list = ec2_client.describe_security_groups()

    return security_group_list['SecurityGroups']

def createSecurityGroup(ec2_client, group_name_expected, security_group_list):
    """Create the desired security group for the environment only if it is not already created

    Args:
        ec2_client (EC2): EC2 client.
        group_name_expected, (string): A name for the security group.
        security_group_list (list): A list of security group

    Returns:
        group_id (string): The id of the created group
    """

    # Set the name and description for the security group
    group_name = group_name_expected
    description = 'Security group for matrixInstanceSecurityGroup resources'

    group_id = ""

    # Check if a security group with the name "matrixInstanceSecurityGroup" is present
    group_exists = False
    for group in security_group_list:
        if group['GroupName'] == group_name:
            group_id = group['GroupId']
            group_exists = True
            break

    # If the security group does not exist, create it
    if not group_exists:
        response = ec2_client.create_security_group(GroupName=group_name, Description=description)
        group_id = response['GroupId']
        # Authorize the SSH connection
        response = ec2_client.authorize_security_group_ingress(
            GroupName=group_name,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [
                        {
                            'CidrIp': '0.0.0.0/0'
                        }
                    ]
                },
            ]
        )

        # Authorize the HTTP connection
        response = ec2_client.authorize_security_group_ingress(
            GroupName=group_name,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 80,
                    'ToPort': 80,
                    'IpRanges': [
                        {
                            'CidrIp': '0.0.0.0/0'
                        }
                    ]
                },
            ]
        )

        # Authorize the TCP connection
        response = ec2_client.authorize_security_group_ingress(
            GroupName=group_name,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 8501,
                    'ToPort': 8501,
                    'IpRanges': [
                        {
                            'CidrIp': '0.0.0.0/0'
                        }
                    ]
                },
            ]
        )
        print(f'Security group {group_name} created')
    else:
        print(f'Security group {group_name} already exists')

    return group_id


##################################################### SQS ##############################################################

def createQueues(sqs_client, sqs_resource, queues_list):
    """Create the desired queues for the environment only if they are not already created

    Args:
        sqs_client (SQS): SQS client.
        sqs_resource (SQS_resource) SQS resource.
        queues_list (list[string]): A list queues name
    """
    for queueName in queues_list:
        try:
            # Get the queue
            queue = sqs_resource.get_queue_by_name(QueueName=queueName)
            print(f'Queue already created with URL: {queue}')
        except ClientError as e:
            # Create the queue
            response = sqs_client.create_queue(QueueName=queueName)
            # Get the URL of the queue
            queue_url = response['QueueUrl']
            print(f'Queue created with URL: {queue_url}')


########################################################################################################################

if __name__ == "__main__":
    ##### S3 #####

    # Init S3 client
    s3 = initClient('s3')
    # Get the list of all buckets already created
    buckets_list = getBucketsList(s3)
    # Create the bucket if it is not already created
    createBucket(s3, shared.S3_NAME, buckets_list)
    # Fill the S3 directory with the files from the local directory
    fillBucket(s3, shared.LOCAL_DIR_TO_COPY, shared.S3_NAME)

    ##### Security Group #####

    # Init EC2 client
    ec2 = initClient('ec2')
    # Get all the security group
    security_group_list = getAllSecurityGroup(ec2)
    # Create the security group if it is not already created
    security_group_id = createSecurityGroup(ec2, shared.GROUP_NAME, security_group_list)

    ##### EC2 #####

    # Get the list of all instances
    instance_list = getAllInstance(ec2)
    # For each instance expected, start instance if it is already created otherwise create it
    initAllInstances(ec2, instance_list, shared.INSTANCE_LIST_NAME, security_group_id)

    ##### SQS #####

    # Init SQS resource
    sqs_r = initResource('sqs')
    # Init SQS client
    sqs_c = initClient('sqs')
    # Create expected queues if they do not exist
    createQueues(sqs_c, sqs_r, shared.QUEUE_LIST_NAME)


    print("Environment successfully initialise")
