from src.manager import Maneger


name="trial"
mode="train"
repositories = [
    {
        "name": "cassandra",
        "url": "https://github.com/apache/cassandra.git",
        "CommitTarget": "ee8803900b5ad4ffe9b827c64e9cab1d4b8ce499",
        "filterFile": "",#r".*src\\java\\org\\apache\\cassandra\\service\\WriteResponseHandler.java",
        "codeIssueJira": "CASSANDRA",
        "projectJira": "issues.apache.org/jira",
    }
]
parameters = {}
option = {
    "name": name,
    "mode": mode,
    "repositories": repositories,
    "parameters": parameters #needless when to infer.
}



maneger = Maneger(option)
maneger.do()