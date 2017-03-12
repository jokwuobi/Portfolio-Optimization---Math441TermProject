# Math441TermProject
Portfolio Optimization in Python using ideas from quadratic and non-linear  programming

By: Judah Okwuobi, Rohin Patel, Brandon Loss

Code for our term project in Mathh 441 (Mathematical Modeling: Discrete Optimization Problems)

Pulling ticker names from the Nasdaq website, we store all nasdaq and Nyse stocks in a database and then filter 
the stocks to those having full data over the range we're interested in (2008-2016 Post recession period). Using
a convex optimization package (Cvxopt) we implement a quadratic programming model to select an optimal portfolio 
over the periods in consideration. We then extend the analysis to allow for borrowing, to include transaction 
costs, and to allow for periodic rebalancing at multiple frequencies (Weekly, Monthly, Quarterly, Annually).
