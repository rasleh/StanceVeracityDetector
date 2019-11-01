This README describes the content of the DAST folder. 

It contains the following items: 
- the dataset, contained in dataset/ 
- a data statement, contained in DAST_data_statement.pdf.
- annotations/ containing two files providing an overview of all stance and rumour veracity annotations:
	- comment_sdqc_stance.txt contains an overview of the SDQC stance for each comment, the first column being the submission IDs and the second being either 0(Supporting), 1(Denying), 2(Querying), or 3(Commenting).
	- A total of 16 submissions were deemed rumourous, and an overview is provided in rumour_overview.txt, including submission ID, rumour veracity, and title. 0 denotes false, 1 denotes true, and 2 denotes unverified.

The dataset is subdivided into 11 folders, one for each subject.
Within each subject-folder, a number of JSON files contain data on independent Reddit submissions within that particular subject.
The JSON files are named after the submission IDs, e.g. "76y6rb" which refers to "Björks FB post om Lars Von Trier (#MeToo)" and is located at https://www.reddit.com/r/Denmark/comments/76y6rb/.

The dataset, DAST, includes 3,007 SDQC-annotated Danish Reddit posts across 33 submissions, or 1,161 branches.
The JSON submission data contained within each JSON file is structured as below, with two objects, one for submission info and another with the lists of branches within that submission.
The example has default entry values. Note that the SDQC annotations are "SourceSDQC" for the stance of the source post, and respectively "SDQC_Submission" and "SDQC_Parent" for each comment.

{
    "redditSubmission": {
        "submission_id": "",
        "title": "",
        "text": "",
        "created": "",
        "num_comments": 0,
        "url": "",
        "text_url": "",
        "upvotes": 0,
        "is_video": false,
        "user": {
            "id": "",
            "created": "",
            "karma": 0,
            "gold_status": false,
            "is_employee": false,
            "has_verified_email": false
        },
        "subreddit": null,
        "comments": null,
        "IsRumour": false,
        "TruthStatus": "",
        "RumourDescription": "",
        "SourceSDQC": ""
    },
    "branches": [
        [
            {
                "comment": {
                    "comment_id": "",
                    "text": "",
                    "parent_id": "",
                    "comment_url": "",
                    "created": "",
                    "upvotes": 0,
                    "is_submitter": false,
                    "is_deleted": false,
                    "replies": 0,
                    "user": { ... },
                    "submission_id": "",
                    "SDQC_Submission": "",
                    "SDQC_Parent": "",
                    "Certainty": "",
                    "Evidentiality": "",
                    "AnnotatedAt": ""
                }
            },
            ...
        ],
        ...
    ]
}


The submissions in the dataset is also annotated for rumours as being either true, false, or unverified, which is noted in the "TruthStatus" entry in the submission JSON object.
Out of the 16 rumourous submissions, three are true, three are false and the rest are unverified.
The rumours make up 220 Reddit conversations, or 596 branches, with a total of 1,489 posts, equal to about half of the dataset. 