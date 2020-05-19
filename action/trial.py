from src.intermediate import intermediate
identifiersDataset=[]
parameterTrain={}
identifierDataset={
    "urlRepository":"https://github.com/apache/cassandra.git",
    "nameRepository":"cassandra",
    "IdCommit":"ee8803900b5ad4ffe9b827c64e9cab1d4b8ce499",
    "filterFile":"",
    "codeIssueJira":"CASSANDRA",
    "projectJira":"issues.apache.org/jira",
}
identifiersDataset.append(identifierDataset)

intermediate("trial", identifiersDataset, parameterTrain)