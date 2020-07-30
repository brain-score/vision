## Submission system

What to do...
### ...when changing the database schema
When you change the database schema, test your new code on the dev database in postgres.
When you change the schema, you also have to adjust the django `models.py` in project brain-score.web.
When creating a new schema, submission system tests will fail. 