import boto3
import shared
from botocore.exceptions import ClientError


def initClient(element_name):
    client = boto3.client(element_name)
    return client


def getInstanceIds(ec2_client, instance_expected_list):
    instance_ids = []
    for name in instance_expected_list:
        instances = ec2_client.describe_instances(
            Filters=[
                {
                    'Name': 'tag:Name',
                    'Values': [name]
                },
                {
                    'Name': 'instance-state-name',
                    'Values': ['running']
                }
            ]
        )['Reservations']

        instance_ids += [instance['Instances'][0]['InstanceId'] for instance in instances]

    return instance_ids


def stopInstance(instance_ids):
    if instance_ids:
        # Do a dryrun first to verify permissions
        try:
            ec2.stop_instances(InstanceIds=instance_ids, DryRun=True)
        except ClientError as e:
            if 'DryRunOperation' not in str(e):
                raise

        # Dry run succeeded, call stop_instances without dryrun
        try:
            response = ec2.stop_instances(InstanceIds=instance_ids, DryRun=False)
            print(f"Successfully stopped {len(instance_ids)} instances")
        except ClientError as e:
            print(e)


if __name__ == "__main__":
    # Init client
    ec2 = initClient('ec2')
    # Get the id of all the expected instances running
    instance_ids = getInstanceIds(ec2, shared.INSTANCE_LIST_NAME)
    # Stop all instances
    stopInstance(instance_ids)
