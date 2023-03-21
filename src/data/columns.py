# my first selection
columns_1 = [
    'Average Synchronous Single-Block Read Latency',
    'Background CPU Usage Per Sec',
    'Background Checkpoints Per Sec',
    'Background Time Per Sec',
    'Branch Node Splits Per Sec',
    'Branch Node Splits Per Txn',
    'Buffer Cache Hit Ratio',
    'Consistent Read Changes Per Sec',
    'Consistent Read Changes Per Txn',
    'Consistent Read Gets Per Sec',
    'Consistent Read Gets Per Txn',
    'CPU Usage Per Sec',
    'CPU Usage Per Txn',
    'Cursor Cache Hit Ratio',
    'DB Block Changes Per Sec',
    'DB Block Changes Per Txn',
    'DB Block Gets Per Sec',
    'DB Block Gets Per Txn',
    'Database CPU Time Ratio',
    'Database Time Per Sec',
    'Database Wait Time Ratio',
    'Enqueue Deadlocks Per Sec',
    'Enqueue Deadlocks Per Txn',
    'Execute Without Parse Ratio',
    'Hard Parse Count Per Sec',
    'Hard Parse Count Per Txn',
    'Leaf Node Splits Per Sec',
    'Leaf Node Splits Per Txn',
    'Logical Reads Per Sec',
    'Long Table Scans Per Sec',
    'Memory Sorts Ratio',
    'Open Cursors Per Sec',
    'PGA Cache Hit %',
    'Physical Reads Per Sec',
    'Physical Reads Per Txn',
    'Physical Reads Per Sec',
    'Physical Reads Per Txn',
    'Physical Write Bytes Per Sec',
    'Physical Write IO Requests Per Sec',
    'Physical Writes Per Sec',
    'Physical Writes Per Txn',
    'Process Limit %',
    'PX downgraded to serial Per Sec',
    'PX operations not downgraded Per Sec',
    'Recursive Calls Per Sec',
    'Redo Allocation Hit Ratio',
    'Redo Generated Per Sec',
    'Redo Generated Per Txn',
    'Response Time Per Txn',
    'Row Cache Hit Ratio',
    'Row Cache Miss Ratio',
    'Run Queue Per Sec',
    'SQL Service Response Time',
    'Session Count',
    'Shared Pool Free %',
    'Total Table Scans Per Sec',
    'Total Table Scans Per Txn',
    'User Calls Ratio',
    'User Commits Percentage',
    'User Limit %',
    'User Rollbacks Percentage',
    'User Transaction Per Sec'
]

# columns used by Gregorio
columns_2 = [
    'Physical Reads Per Sec',
    'Physical Reads Per Txn',
    'Physical Writes Per Sec',
    'Physical Writes Per Txn',
    
    'Hard Parse Count Per Sec',
    'Hard Parse Count Per Txn',
    
    # 'Executions Per Sec',
    # 'Executions Per Txn',
    
    'Long Table Scans Per Sec',
    'Long Table Scans Per Txn',
    'Total Table Scans Per Sec',
    'Total Table Scans Per Txn',
    
    'Library Cache Hit Ratio',
    'Library Cache Miss Ratio',
    
    'Redo Generated Per Sec',
    'Redo Generated Per Txn',
    'Redo Writes Per Sec',
    'Redo Writes Per Txn',
    
    'Cursor Cache Hit Ratio',
    
    'I/O Megabytes per Second',
    # 'I/O Requests per Second',  # strange behaviour difficult to predict
    
    'Process Limit %',
    
    'CPU Usage Per Sec',
    'CPU Usage Per Txn',
    'Host CPU Utilization (%)',
    'Background CPU Usage Per Sec',
    'Database Wait Time Ratio',
    'Database CPU Time Ratio',
    
    'Database Time Per Sec'
]

# first list of metrics considered in a 2006 Oracle patent
columns_3 = [
    'SQL Service Response Time',
    'Response Time Per Txn',
    'Database Time Per Sec',
     # 'Workload Volume Metrics',  # not existing
    'User Transaction Per Sec',
    'Physical Reads Per Sec',
    'Physical Writes Per Sec',
    'Redo Generated Per Sec',
    'User Calls Per Sec',
    'Network Traffic Volume Per Sec',
    'Current Logons Count',
    'Executions Per Sec',
    'Logical Reads Per Txn',
    'Total Parse Count Per Txn',
    'Enqueue Requests Per Txn',
    'DB Block Changes Per Txn',
    
    'Database Time Per Sec'
]

# second list of metrics considered in a 2006 Oracle patent
columns_4 = [
    'Buffer Cache Hit Ratio',
    'User Transaction Per Sec',
    'Physical Reads Per Sec',
    'Physical Writes Per Sec',
    'Redo Generated Per Txn',
    'Logons Per Sec',
    'Logons Per Txn',
    'User Commits Per Sec',
    'User Rollbacks Percentage',
    'User Calls Per Sec',
    'User Calls Per Txn',
    'Logical Reads Per Txn',
    'Redo Writes Per Sec',
    'Total Parse Count Per Sec',
    'Total Parse Count Per Txn',
    'Cursor Cache Hit Ratio',
    'Execute Without Parse Ratio',
    'Host CPU Utilization (%)',
    'Network Traffic Volume Per Sec',
    'Enqueue Requests Per Txn',
    'DB Block Changes Per Txn',
    'CPU Usage Per Sec',
    'CPU Usage Per Txn',
    'Current Logons Count',
    'SQL Service Response Time',
    'Database Wait Time Ratio',
    'Database CPU Time Ratio',
    'Response Time Per Txn',
    'Executions Per Txn',
    'Executions Per Sec',
    
    'Database Time Per Sec'
] 

columns_5 = set(columns_3 + columns_4)

columns_6 = [
    'Average Active Sessions',
    'CPU Usage Per Sec',
    'Consistent Read Gets Per Sec',
    'DB Block Changes Per Sec',
    'DBWR Checkpoints Per Sec',
    'Enqueue Waits Per Sec',
    'Executions Per Sec',
    'Hard Parse Count Per Sec',
    'Host CPU Usage Per Sec',
    'I/O Megabytes per Second',
    'Logical Reads Per Sec',
    'Logons Per Sec',
    'Physical Reads Per Sec',
    'Physical Writes Per Sec',
    'Redo Generated Per Sec',
    'User Calls Per Sec',
    'User Commits Per Sec',
    'User Rollbacks Per Sec',
    'Database Time Per Sec'
]

columns_7 = [
    'Average Active Sessions',
    'Executions Per Sec',
    'Database Time Per Sec'
]


def get_columns_list(columns_name='columns_2'):
    if columns_name == 'columns_1':
        return columns_1
    elif columns_name == 'columns_2':
        return columns_2
    elif columns_name == 'columns_3':
        return columns_3
    elif columns_name == 'columns_4':
        return columns_4
    elif columns_name == 'columns_5':
        return columns_5
    elif columns_name == 'columns_6':
        return columns_6
    elif columns_name == 'columns_7':
        return columns_7
    else:
        print('Error: invalid column name')
        print('\treturning default metrics')
        return columns_2