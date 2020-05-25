from src.manager import Maneger


name="trial"
mode="train"
repositories = [
    {
        "name": "cassandra",
        "url": "https://github.com/apache/cassandra.git",
        "CommitTarget": "df724579efeee15a8974e83be07462a9574b8ae3",#c206ca0bd64355eef5d992afcf28ec698f0a4f85
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