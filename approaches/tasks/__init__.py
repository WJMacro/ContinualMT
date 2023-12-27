import os
from fairseq.tasks import import_tasks


# automatically import any Python files in the tasks/ directory
tasks_dir = os.path.dirname(__file__)
import_tasks(tasks_dir, "approaches.tasks")
