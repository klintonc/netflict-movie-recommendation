# Netflict Movie Recommendation

ITCS208 Object Oriented Programming — Project 3, Faculty of ICT, Mahidol University (MUICT).

## What it is

A Java movie recommender based on user-based collaborative filtering, using MovieLens-style data. The program:

- parses movie and rating CSV files into `Movie`, `User` and `Rating` objects
- trains a model: a user–movie rating matrix plus a user–user similarity matrix, saved to a model file
- recommends the top-K unseen movies for a user, within a range of release years
- evaluates the recommender on a test set using RMSE, correlation, precision, recall and F1

`BaseMovieRecommender` defines the interface and `SimpleMovieRecommender` implements it. `MainController` is the command-line entry point and `Evaluator` computes the metrics.

### Datasets

| Folder  | Contents |
|---------|----------|
| `micro` | Small sample data with an example model and result file |
| `q1`, `q2` | Test cases |
| `small` | Main dataset, plus `tags.train.csv` for bonus features |

## How to run

You need a JDK (8 or newer). Dependencies are in `lib/`.

```bash
# compile
mkdir -p out
javac -cp "lib/*" -d out *.java

# test parsing
java -cp "out:lib/*" MainController parse micro/movies.csv micro/users.train.csv

# train a model
java -cp "out:lib/*" MainController train small/movies.csv small/users.train.csv small/small.simple.model

# evaluate
java -cp "out:lib/*" MainController test small/movies.csv small/users.train.csv small/small.simple.model small/users.test.csv small/small.simple.result

# recommend movies for user 19
java -cp "out:lib/*" MainController recommend small/movies.csv small/users.train.csv small/small.simple.model 19
```

On Windows, use `;` instead of `:` in the classpath.
