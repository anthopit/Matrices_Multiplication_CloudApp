import numpy as np
import ast
import boto3
from botocore.config import Config


def initSQS(QeueName):

    config = Config(
        region_name='us-east-1',
        signature_version='v4',
        retries={
            'max_attempts': 10,
            'mode': 'standard'
        }
    )
    # Get the service resource
    sqs = boto3.resource('sqs', config=config)

    # Get the queue
    queue = sqs.get_queue_by_name(QueueName=QeueName)

    return queue


def sendResult(queue, message):
    # Create a new message
    response = queue.send_message(MessageBody=message)

    # The response is NOT a resource, but gives you a message ID and MD5
    print(response.get('MessageId'))


def encodeMessageWorker(index, shape, m):
    index_str = str(index)
    shape_str = str(shape)
    m_str = str(m.tolist())

    result = [index_str, shape_str, m_str]
    return '/'.join(result)


def decodeMessageWorker(message):
    message_recv = message.split('/')

    type = message_recv[0]
    index = ast.literal_eval(message_recv[1])
    shape = ast.literal_eval(message_recv[2])
    m1_list = ast.literal_eval(message_recv[3])
    m2_list = ast.literal_eval(message_recv[4])
    m1 = np.array(m1_list, dtype=int)
    m2 = np.array(m2_list, dtype=int)

    return type, index, shape, m1, m2


def process_message(message_body):
    type, index, shape, m1, m2 = decodeMessageWorker(message_body)

    print('Index: ', index)
    print('Shape: ', shape)
    print('M1: ', m1)
    print('M2: ', m2)
    print('**********************')

    if type == '1':
        result = np.dot(m1, m2)
    else:
        result = m1 + m2

    # modif ici
    return index, shape, result


if __name__ == "__main__":
    queue_receive = initSQS('matrixQueueMtoW')
    queue_send = initSQS('matrixQueueWtoM')

    count = 0

    while True:
        messages = queue_receive.receive_messages()
        for message in messages:
            count += 1
            index, shape, result = process_message(message.body)
            result_message = encodeMessageWorker(index, shape, result)
            sendResult(queue_send, result_message)
            print(f"Total send: {count}", end="\r")
            message.delete()
