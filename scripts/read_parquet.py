from pyspark.sql import SparkSession

"""
The 10 MedRAG PubMed parquet files can be downloaded one at a time by click on the download button at
trhttps://huggingface.co/datasets/MedRAG/pubmed/tree/refs%2Fconvert%2Fparquet/default/partial-train
It takes only about 10 minutes.
Each file is between 200 and 300 MBs. And have between 200000 and 263000 articles each.
All 10 files are 2.78 GB. They include approximately 2.2 million articles.
After extracting the abstracts (i.e. "content"), the files halve in size,  at about 125 MB each.

In total all 10, abstracts only, are 1.26 GB. 
SO, THAT INCLUDES ALL OF THE ARTICLES IN MEDRAG_PUBMED, ABSTRACTS ONLY.  
"""

if __name__ == '__main__':

    for parquet_file_number in range(1, 10):
        spark = SparkSession.builder.appName('pubmed_parquets').getOrCreate()
        parquet_file_name = '{:04d}.parquet'.format(parquet_file_number)
        parquet_file_path = f'../dataset/PUBMED/{parquet_file_name}'
        sdf = spark.read.parquet(parquet_file_path)
        sdf.show()
        print(f'Number of rows in {parquet_file_name} dataframe = {sdf.count()}')
        # Number of rows in 0000.parquet dataframe = 256634
        # Number of rows in 0001.parquet dataframe = 262939
        # Number of rows in 0002.parquet dataframe = 205881
        # Number of rows in 0003.parquet dataframe = 205315
        # Number of rows in 0004.parquet dataframe = 213527
        # Number of rows in 0005.parquet dataframe = 216866
        # Number of rows in 0006.parquet dataframe = 214314
        # Number of rows in 0007.parquet dataframe = 212964
        # Number of rows in 0008.parquet dataframe = 217815
        # Number of rows in 0009.parquet dataframe = 203584

        # Extract abstracts and save to parquets
        abstracts_sdf = sdf.select('content')
        abstracts_sdf.show()
        abstracts_parquet_file_name = '{:04d}.parquet'.format(parquet_file_number)
        output_parquet_file_path = f'../dataset/PUBMED/MedRAG_abstracts/{abstracts_parquet_file_name}'
        abstracts_sdf.write.parquet(output_parquet_file_path)
