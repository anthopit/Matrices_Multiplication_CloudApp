import numpy as np
from math import *
import boto3
import ast
import time
import multiprocessing as mp
from multiprocessing import shared_memory
from botocore.config import Config

def initSQS(QeueName):
    """Return a SQS queue interface of the queue with the specified name.

    Args:
        QeueName (string): qeue name

    Returns:
        queue (SQS) : SQS queue
    """
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

def sendJob(message):
    # Create a new message
    response = queue_send.send_message(MessageBody=message)


def encodeMessageMaster(index,shape,m1,m2, isMultiplication):
    """Create the message to send with the good communicqtion protocole by converting each element in string and gather them.

    Args:
        index (list): qeue name
        shape (list): shape oh the matrices
        m1 (numpy array): first numpy sub-matrix
        m2 (numpy array): second numpy sub-matrix
        isMultiplication (boolean): Indicator for type of calculation

    Returns:
        message
    """
    type = '0'

    if isMultiplication:
        type = '1'

    index_str = str(index)
    shape_str = str(shape)
    m1_str = str(m1.tolist())
    m2_str = str(m2.tolist())
    #m1_str = np.array_str(m1)
    #m2_str = np.array_str(m2)

    result = [type,index_str,shape_str,m1_str,m2_str]
    return '/'.join(result)

def decodeMessageMaster(message):
    """Convert each element from sting message with the protocol communication

    Args:
        message (string): message from worker

    Returns:
        index (lis): Index of the result matrix
        m (numpy array): Result matrix
    """
    messageRecv = message.split('/')

    index = ast.literal_eval(messageRecv[0])
    shape = ast.literal_eval(messageRecv[1])
    m_list = ast.literal_eval(messageRecv[2])
    m = np.array(m_list, dtype=int)

    return index, m

def generateMatrix(x, y):
    # Return a matrix of size x.y with values between 0 and 100
    m = np.random.randint(100, size=(x, y))
    return m



def devideMatrix(M, subMatrixSize):
    """Devide a matrix into a stack of sub-matrices

    Args:
        M (numpy array): matrix to divide
        subMatrixSize (int) : Number of element per sub-matrices

    Returns:
        paddindX (int): Padding added on the x-axis
        paddindY (int): Padding added on the y-axis
        M (numpy array): Devided matrix
    """
    l_sub_matrix = ceil(sqrt(subMatrixSize))

    # Define the necessary padding
    paddingX = (ceil(M.shape[1] / l_sub_matrix)*l_sub_matrix) - M.shape[1]
    paddingY = (ceil(M.shape[0] / l_sub_matrix)*l_sub_matrix) - M.shape[0]

    # Add a padding
    M = np.pad(M, ((0, paddingY), (0, paddingX)))

    # Devision of the matrix
    M = np.lib.stride_tricks.as_strided(M, shape=(int(M.shape[0] / l_sub_matrix), int(M.shape[1] / l_sub_matrix), l_sub_matrix, l_sub_matrix), strides=(M.shape[1] * l_sub_matrix, l_sub_matrix, M.shape[1], 1))


    return paddingX, paddingY, M

def processMessage(message_body, final_matrix, isMultiplication):
    index, m = decodeMessageMaster(message_body)

    if isMultiplication:
        # Add the received matrix to the final matrix
        final_matrix[index[0]][index[1]] += m
    else:
        # Replace the received matrix to the final matrix
        final_matrix[index[0]][index[1]] = m

def getTotalToReceive(m1, m2, isMultiplication):
    """Compute the number of packets to receive

    Args:
        m1 (numpy array): first matrix
        m1 (numpy array): second matrix
        isMultiplication (boolean): Indicator for type of calculation

    Returns:
        total_to_receive (int): Number of packet to receive
    """
    if isMultiplication:
        total_to_receive = M1.shape[0] * M2.shape[0] * M1.shape[1]
    else:
        total_to_receive = m1.shape[0] * m1.shape[1]

    return total_to_receive

def shapeFinalMatrix(matrix, final_shape):
    # Gather the sub-matrices to create the final matrix
    matrix = np.concatenate(matrix, axis=1)
    matrix = np.concatenate(matrix, axis=1)
    matrix = matrix[:final_shape[0], :final_shape[1]]

    return

def sendMatrix(M1, M2, total, isMultiplication):
    """Run through the two matrices M1 and M2 to send the packet pairs

    Args:
        M1 (numpy array): first matrix
        M2 (numpy array): second matrix
        total (int): total number of packets to send
        isMultiplication (boolean): Indicator for type of calculation
    """
    count = 0

    if isMultiplication :
        index1 = [0, 0]
        index2 = [0, 0]


        for y2 in M2:
            for x1 in M1:
                index1[1] = 0
                index2[0] = 0
                for x, y in zip(x1, y2):
                    sendJob(encodeMessageMaster([index1[0],index2[1]],np.shape(x),x,y.T, isMultiplication))
                    count += 1
                    print(f"Total send: {count}/{total}", end="\r")
                    index1[1] += 1
                    index2[0] += 1
                index1[0] += 1
            index1[0] = 0
            index2[1] += 1
            
    else:
        index = [0,0]

        for X1,X2 in zip(M1,M2):
            index[1] = 0
            for x1,x2 in zip(X1, X2):
                sendJob(encodeMessageMaster(index,np.shape(x1),x1,x2, isMultiplication))
                index[1] += 1
            index[0] += 1
        
def receiveMatrix(total_to_receive, final_matrix, isMultiplication):
    number_receive = 0

    while number_receive < total_to_receive:
        print(f"Total receive: {number_receive}/{total_to_receive}", end="\r")
        messages = queue_receive.receive_messages()
        for message in messages:
            number_receive += 1
            processMessage(message.body, final_matrix, isMultiplication)
            message.delete()


if __name__ == "__main__":
    # Define the type of computation
    isMultiplication = True

    # init the SQS queues
    queue_receive = initSQS('matrixQueueWtoM')
    queue_send = initSQS('matrixQueueMtoW')

    # Define the number of elements per sub-matrices
    SUBMATRIX_ELEMENT_NUMBER = 30000

    # Generate the two matrices
    M1 = generateMatrix(1000, 1000)
    M2 = generateMatrix(1000, 1000)

    # init the shape of final matrix
    M_shape=[M1.shape[0], M2.shape[1]]

    print(M1)
    print(M2)

    # Classic multiplicqtion and addition of matrices
    time_1 = time.time()
    if isMultiplication:
        print(np.dot(M1, M2))
    else:
        print(M1 + M2)
    print("Execution time : ", (time.time() - time_1))
    print("*************************************************")


    time_2 = time.time()

    if isMultiplication:
        # Transpose of M2
        M2 = np.transpose(M2)
        # Necessary to replace the byte in good order
        M2 = M2.tolist()
        M2 = np.array(M2)

    # Define the type of the matrices
    M1 = M1.astype('int8')
    M2 = M2.astype('int8')

    # Devide the matrices
    padding11, padding12, M1 = devideMatrix(M1, SUBMATRIX_ELEMENT_NUMBER)
    padding21, padding22, M2 = devideMatrix(M2, SUBMATRIX_ELEMENT_NUMBER)

    # get the number total of packets to receive
    total_to_receive = getTotalToReceive(M1, M2, isMultiplication)

    # init a final matrix fill with 0
    final_matrix = np.zeros([M1.shape[0], M2.shape[0], M1.shape[2], M1.shape[2]], dtype=int)

    # Define qnd lauch the first process
    process1 = mp.Process(target=sendMatrix, args=(M1, M2, total_to_receive, isMultiplication))
    process1.start()

    # Create a shared memory for the final matrix
    shm = shared_memory.SharedMemory(create=True, size=final_matrix.nbytes)
    # Now create a NumPy array backed by shared memory
    final_matrix_buffer = np.ndarray(final_matrix.shape, dtype=final_matrix.dtype, buffer=shm.buf)
    final_matrix_buffer[:] = final_matrix[:]  # Copy the original data into shared memory

    # Define and lauch the second process
    process2 = mp.Process(target=receiveMatrix, args=(total_to_receive, final_matrix_buffer, isMultiplication))
    process2.start()

    process1.join()
    process2.join()

    # Copy the final matrix from the shared memory to the final matrix from the main process
    final_matrix[:] = final_matrix_buffer[:]

    # Clean up from within the first Python shell
    del final_matrix_buffer  # Unnecessary; merely emphasizing the array is no longer used
    shm.close()
    shm.unlink()  # Free and release the shared memory block at the very end

    final_matrix = np.concatenate(final_matrix, axis=1)
    final_matrix = np.concatenate(final_matrix, axis=1)
    final_matrix = final_matrix[:M_shape[0], :M_shape[1]]

    print(final_matrix)

    print("Execution time : ", (time.time() - time_2))


