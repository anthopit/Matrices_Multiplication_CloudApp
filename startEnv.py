import boto3
import shared
import paramiko

key = paramiko.RSAKey.from_private_key_file(shared.LABSUSER_PATH)
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ec2 = boto3.client('ec2')

for name in shared.INSTANCE_LIST_NAME:
    target_instances = ec2.describe_instances(
        Filters=[
            {
             'Name':'tag:Name',
             'Values':[name]
            },
    {
                'Name': 'instance-state-name',
                'Values': ['running']
            }
        ]
    )['Reservations']

    try:
        public_ip = target_instances[0]['Instances'][0]['PublicIpAddress']
        print(public_ip)

        client.connect(hostname=public_ip, username='ec2-user', pkey=key)

        if name != 'master':
            stdin, stdout, stderr = client.exec_command('aws s3 cp s3://matrixs3operfkemngotnspoff/workerCode.py code/workerCode.py')
            stdin, stdout, stderr = client.exec_command('nohup python3 code/workerCode.py > nohup.out 2> nohup.err < /dev/null &')
            print(f'{name} is running.')
        else:
            stdin, stdout, stderr = client.exec_command('aws s3 cp s3://matrixs3operfkemngotnspoff/masterCode.py code/masterCode.py')
            stdin, stdout, stderr = client.exec_command('aws s3 cp s3://matrixs3operfkemngotnspoff/appStreamLit.py code/appStreamLit.py')
            stdin, stdout, stderr = client.exec_command('streamlit run code/appStreamLit.py')

            def line_buffered(f):
                line_buf = ""
                while not f.channel.exit_status_ready():
                    line_buf += f.read(1).decode('utf-8')
                    if line_buf.endswith('\n'):
                        yield line_buf
                        line_buf = ''


            for l in line_buffered(stdout):
                print(l)

    except:
        print("Error: the EC2 instance is not running : please run initEnv.py before.")


# Close the client
client.close()