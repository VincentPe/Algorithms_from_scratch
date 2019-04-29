# Algorithms from scratch
I started following tutorials on writing machine learning algorithms from scratch in python,
as it has become too easy to just import and use packages without knowing how exactly a ML-algo
comes about its predictions. <br>
I noticed that when comparing the results of different machine learning models for Kaggle cases
or work problems, I sometimes did not understand why exactly one performed better than the other. <br>
I have combined resources from different tutorials in a way that I myself understand them,
but perhaps they might be useful to other people too. <br>

## Technical description
Initially this project started as one python file, scheduled on a remote desktop, but the monitoring took a lot of work.
Due to the many dependencies in the project on external sources like the DWH, Transcillary and the internal API's,
often the project did not run and it was time consuming to find the changing source of the problem.
Using Airflow, the project can run in a more stable environment and parts of the project can already run while others need
to wait for dependencies to complete. The UI makes debugging easier and provides trends in errors.
The logs can function as input for dashboarding, etc.

## Additional info
As mentioned earlier, I have gathered information from multiple websites per algo. <br>
I have tried to keep track of resources and stated them in the notebooks, but might have missed some
or have just forgotten in some cases.

## Structure


## Usage
There is great documentation on https://airflow.apache.org/ on how to use Airflow.
Most important after installation is to use `airflow initdb` to initialize a database.
Then from the terminal use `airflow webserver -p <portnumber>` and tunnel the port from the Jupyter Hub's server to your local machine using e.g. Putty.
The UI should now be visible in the browser using `http://localhost:<portnumber>/admin/`.
After that, open another terminal and start the scheduler: `airflow scheduler`, which will start running tasks that are 'ON'.
