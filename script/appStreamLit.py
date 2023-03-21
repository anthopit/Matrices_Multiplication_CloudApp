import boto3
import streamlit as st
import numpy as np
from math import *
import ast
import time
import multiprocessing as mp
from multiprocessing import shared_memory, Manager
from botocore.config import Config
from ctypes import c_char_p


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


def sendJob(message):
    # Create a new message
    response = queue_send.send_message(MessageBody=message)


def encodeMessageMaster(index, shape, m1, m2, isMultiplication):
    type = '0'

    if isMultiplication:
        type = '1'

    index_str = str(index)
    shape_str = str(shape)
    m1_str = str(m1.tolist())
    m2_str = str(m2.tolist())
    # m1_str = np.array_str(m1)
    # m2_str = np.array_str(m2)

    result = [type, index_str, shape_str, m1_str, m2_str]
    return '/'.join(result)


def decodeMessageMaster(message):
    messageRecv = message.split('/')

    index = ast.literal_eval(messageRecv[0])
    shape = ast.literal_eval(messageRecv[1])
    m_list = ast.literal_eval(messageRecv[2])
    m = np.array(m_list, dtype=int)

    return index, m


def generateMatrix(x, y):
    m1 = np.random.randint(0, 100, (x, y))
    return m1


def devideMatrix(M, subMatrixSize):
    l_sub_matrix = ceil(sqrt(subMatrixSize))

    # Define the necessary padding
    paddingX = (ceil(M.shape[1] / l_sub_matrix) * l_sub_matrix) - M.shape[1]
    paddingY = (ceil(M.shape[0] / l_sub_matrix) * l_sub_matrix) - M.shape[0]

    # Add a padding
    M = np.pad(M, ((0, paddingY), (0, paddingX)))

    # Devision of the matrix
    M = np.lib.stride_tricks.as_strided(M, shape=(
        int(M.shape[0] / l_sub_matrix), int(M.shape[1] / l_sub_matrix), l_sub_matrix, l_sub_matrix),
                                        strides=(M.shape[1] * l_sub_matrix, l_sub_matrix, M.shape[1], 1))

    return paddingX, paddingY, M


def processMessage(message_body, final_matrix, isMultiplication):
    index, m = decodeMessageMaster(message_body)

    if isMultiplication:
        final_matrix[index[0]][index[1]] += m
    else:
        final_matrix[index[0]][index[1]] = m


def getTotalToReceive(m1, m2, isMultiplication):
    if isMultiplication:
        total_to_receive = m1.shape[0] * m2.shape[0] * m1.shape[1]
    else:
        total_to_receive = m1.shape[0] * m1.shape[1]

    return total_to_receive


def sendMatrix(M1, M2, total, isMultiplication, string_placeholder):

    count = 0

    if isMultiplication:
        index1 = [0, 0]
        index2 = [0, 0]

        for y2 in M2:
            for x1 in M1:
                index1[1] = 0
                index2[0] = 0
                for x, y in zip(x1, y2):
                    sendJob(encodeMessageMaster([index1[0], index2[1]], np.shape(x), x, y.T, isMultiplication))
                    count += 1
                    string_placeholder.value = (f"Total send: {count}/{total}")
                    index1[1] += 1
                    index2[0] += 1
                index1[0] += 1
            index1[0] = 0
            index2[1] += 1

    else:
        index = [0, 0]

        for X1, X2 in zip(M1, M2):
            index[1] = 0
            for x1, x2 in zip(X1, X2):
                sendJob(encodeMessageMaster(index, np.shape(x1), x1, x2, isMultiplication))
                index[1] += 1
            index[0] += 1

def receiveMatrix(total_to_receive, final_matrix, isMultiplication, string_placeholder):
    number_receive = 0

    while number_receive < total_to_receive:
        messages = queue_receive.receive_messages()
        string_placeholder.value = (f"Total received: {number_receive}/{total_to_receive}")
        for message in messages:
            number_receive += 1
            processMessage(message.body, final_matrix, isMultiplication)
            message.delete()


###################################################################################

queue_receive = initSQS('matrixQueueWtoM')
queue_send = initSQS('matrixQueueMtoW')

# ############################ Interface ############################################
st.title("Matrices Computation App")
st.subheader("Select the computation")

type = st.radio(
    "Computation: ",
    ('Addition', 'Multiplication'))

st.subheader("Select the size of the matrices")

isMultiplication = False

if type == 'Addition':
    isMultiplication = False
    number_row_m1 = st.number_input('Number of rows and column matrices', min_value=0)
    number_row_m2 = number_row_m1
    number_column_m1 = number_row_m1
    number_column_m2 = number_row_m1
else:
    isMultiplication = True
    col11, col12 = st.columns(2)
    with col11:
        number_row_m1 = st.number_input('Number of rows in the first matrix', min_value=0)
        number_column_m1 = st.number_input('Number of column in the first matrix', min_value=0)
    with col12:
        number_row_m2 = st.number_input('Number of rows in the second matrix', min_value=0)
        number_column_m2 = st.number_input('Number of column in the second matrix', min_value=0)

st.subheader("Select the of element per submatrices")
submatrix_element_number = st.number_input('Number of element', min_value=0, max_value=30000)

########################################################################################
def runComputation():

    if isMultiplication == False:
        M1 = np.random.randint(100, size=(number_row_m1, number_column_m1))
        M2 = np.random.randint(100, size=(number_row_m2, number_column_m2))

    if isMultiplication:
        if number_column_m1 != number_row_m2:
            st.error(
                'The number of column for the first matrix need to be identical to the number of row of the second matrix.',
                icon="🚨")
        else:
            M1 = np.random.randint(100, size=(number_row_m1, number_column_m1))
            M2 = np.random.randint(100, size=(number_row_m2, number_column_m2))

    col21, col22 = st.columns(2)
    with col21:
        st.subheader("Matrix 1:")
        st.text(str(M1))
    with col22:
        st.subheader("Matrix 2:")
        st.text(str(M2))

    col31, col32 = st.columns(2)
    with col31:
        st.subheader("Serial processing result:")

        with st.spinner('Wait for processing...'):
            time_1 = time.time()

            if isMultiplication:
                st.text(str(np.dot(M1, M2)))
            else:
                st.text(str(M1 + M2))

            st.text("Execution time : ")
            st.text(str((time.time() - time_1)))
        st.success('Done!')

    with col32:
        st.subheader("Parallel processing result:")

        time_2 = time.time()

        if isMultiplication:
            # Transpose of M2
            M2 = np.transpose(M2)
            # Necessary to replace the byte in good order
            M2 = M2.tolist()
            M2 = np.array(M2)

        M1 = M1.astype('int8')
        M2 = M2.astype('int8')

        padding11, padding12, M1 = devideMatrix(M1, submatrix_element_number)
        padding21, padding22, M2 = devideMatrix(M2, submatrix_element_number)

        total_to_receive = getTotalToReceive(M1, M2, isMultiplication)

        manager = Manager()
        string_placeholder1 = manager.Value(c_char_p, "")

        with st.spinner('Send paquet'):
            placeholder1 = st.empty()
            process1 = mp.Process(target=sendMatrix, args=(M1, M2, total_to_receive, isMultiplication, string_placeholder1))
            process1.start()
        st.success('Done!')

        final_matrix = np.zeros([M1.shape[0], M2.shape[0], M1.shape[2], M1.shape[2]], dtype=int)

        shm = shared_memory.SharedMemory(create=True, size=final_matrix.nbytes)
        # Now create a NumPy array backed by shared memory
        final_matrix_buffer = np.ndarray(final_matrix.shape, dtype=final_matrix.dtype, buffer=shm.buf)
        final_matrix_buffer[:] = final_matrix[:]  # Copy the original data into shared memory

        string_placeholder2 = manager.Value(c_char_p, "")

        with st.spinner('Receive paquet'):
            placeholder2 = st.empty()
            process2 = mp.Process(target=receiveMatrix, args=(total_to_receive, final_matrix_buffer, isMultiplication, string_placeholder2))
            process2.start()
        st.success('Done!')

        while process1.is_alive() or process2.is_alive():
            placeholder1.text(string_placeholder1.value)
            placeholder2.text(string_placeholder2.value)

        process1.join()
        process2.join()

        final_matrix[:] = final_matrix_buffer[:]

        # Clean up from within the first Python shell
        del final_matrix_buffer  # Unnecessary; merely emphasizing the array is no longer used
        shm.close()
        shm.unlink()  # Free and release the shared memory block at the very end

        final_matrix = np.concatenate(final_matrix, axis=1)
        final_matrix = np.concatenate(final_matrix, axis=1)
        final_matrix = final_matrix[:number_row_m1, :number_column_m2]
        st.text(str(final_matrix))

        st.text("Execution time : ")
        st.text(str((time.time() - time_2)))

if st.button('Start'):
    runComputation()
