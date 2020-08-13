import json
import logging

from brainscore.submission.utils import get_secret


def connect_db(db_configs):
    import psycopg2
    return psycopg2.connect(host=db_configs['host'], user=db_configs['username'], password=db_configs['password'],
                            dbname=db_configs['dbInstanceIdentifier'])


def store_score(db_configs, score):
    dbConnection = connect_db(db_configs)
    try:
        insert = '''insert into benchmarks_score
                (model, benchmark, score_raw, score_ceiled, error, layer, timestamp, jenkins_job_id, owner, author)   
                VALUES(%s,%s,%s,%s,%s,%s, %s, %s, %s, %s)'''
        logging.info(f'Run results: {score}')
        cur = dbConnection.cursor()
        args = [score['Model'], score['Benchmark'],
                score['raw_result'], score['ceiled_result'],
                score['error'], score['layer'], score['finished_time'],
                score['jenkins_id'], score['email'],
                score['name']]
        cur.execute(insert, args)
        dbConnection.commit()
    finally:
        dbConnection.close()
    return
