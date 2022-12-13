# -*- coding: utf-8 -*-
# Problem Set 5: Experimental Analysis
# Name: 
# Collaborators (discussion):
# Time:

import pylab
import re
import math

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2010)
TESTING_INTERVAL = range(2010, 2016)

class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            
        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d pylab array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return pylab.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.
    
    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by a linear
            regression model
        model: a pylab array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = pylab.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]


def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        degs: a list of degrees of the fitting polynomial

    Returns:
        a list of pylab arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    coeff_degs = []
    for deg in degs:
        coeff_degs.append(pylab.polyfit(x, y, deg))

    return coeff_degs


def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    
    Args:
        y: 1-d pylab array with length N, representing the y-coordinates of the
            N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the R-squared error term
    """
    sse = 0
    for i in range(len(y)):
        sse = sse + (y[i]-estimated[i])**2
        
    mean_y = sum(y)/len(y)
    
    var_y = 0
    for y_i in y:
        var_y = var_y + (y_i - mean_y)**2

    r_squared = 1 - (sse/var_y)

    return r_squared


def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-squared value for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope). 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    for model in models:
        p = pylab.poly1d(model)  # setting degree of polynomial 
        y_estim = p(x)
        r_sq = r_squared(y, y_estim)
        deg_model = len(model) - 1
        
        pylab.figure()                      
        pylab.scatter(x, y, color='blue')   
        pylab.plot(x,y_estim, color='red')
        pylab.xlabel('Years')
        pylab.ylabel('degrees Celsius')

        if deg_model == 1:
            se_slope_ratio = se_over_slope(x, y, y_estim, model)
            # If absolute value of the se_over slope ratio is less than 0.5,
            # the trend is significant (i.e., not by chance).
            pylab.title('Modelling temperature change over the years' + "\n" +
                        "R^2 = " + str(round(r_sq, 2)) + "\n" + "Degree of model = " + str(deg_model) +
                        "\n" + "SE over slope = " + str(se_slope_ratio))
        else:
            pylab.title('Modelling temperature change over the years' + "\n" +
                        "R^2 = " + str(round(r_sq, 2)) + "\n" + "Degree of model = " + str(deg_model))
        pylab.show()


def gen_cities_avg(climate, multi_cities, years):
    """
    Compute the average annual temperature over multiple cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to average over (list of str)
        years: the range of years of the yearly averaged temperature (list of
            int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    y = []
    for year in years:
        yearly_temp = []
        for city in multi_cities:
            yearly_temp.extend(climate.get_yearly_temp(city, year))
        national_yearly_temp_avg = sum(yearly_temp) / len(yearly_temp)
        y.append(national_yearly_temp_avg)

    return y

def moving_average(y, window_length):
    """
    Compute the moving average of y with specified window length.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        window_length: an integer indicating the window length for computing
            moving average

    Returns:
        an 1-d pylab array with the same length as y storing moving average of
        y-coordinates of the N sample points
    """
    mov_avg = []
    # moving average of ​y[i]​ is the average of ​y[i-window_length+1] ​ to y[i]
    for i in range(len(y)):
        count = 0
        window_avg = 0
        for j in range(window_length):
            if i-j >= 0:
                count = count + 1
                window_avg = window_avg + y[i-j]
        window_avg = window_avg/count
        mov_avg.append(window_avg)

    return pylab.array(mov_avg)



def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    sse = 0
    for i in range(len(y)):
        sse = sse + (y[i] - estimated[i]) ** 2

    rmse = math.sqrt(sse/len(y))

    return rmse

'''def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities. 

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual 
        city temperatures for the given cities in a given year.
    """
    avg_yearly_temp = gen_cities_avg(climate, multi_cities, years)

    annual_std = []
    counter = 0
    for year in years:
        all_cities_yearly_temp = []
        for city in multi_cities:
            yearly_temp = climate.get_yearly_temp(city, year)
            all_cities_yearly_temp.append(yearly_temp)
            
        all_cities_yearly_temp = pylab.array(all_cities_yearly_temp)
        daily_mean = all_cities_yearly_temp.mean(axis=0)  # calculating mean for each day from all the city arrays.
        std_dev = pylab.std(daily_mean)  # calculating standard deviation across the year. 
        annual_std.append(std_dev)
    annual_std = pylab.array(annual_std)  # converting list to an array.

    return annual_std'''

def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities.
    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)
    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual
        city temperatures for the given cities in a given year.
    """
    annual_std_dev = []
    for year in years:
        all_cities_yearly_temp = []
        for city in multi_cities:
            yearly_temp = climate.get_yearly_temp(city, year)
            all_cities_yearly_temp.append(yearly_temp)
        all_cities_yearly_temp = pylab.array(all_cities_yearly_temp)  # converting list to an array.
        daily_mean = all_cities_yearly_temp.mean(axis=0)  # calculating mean for each day from all the city arrays.
        std_dev = pylab.std(daily_mean)  # calculating standard deviation across the year.
        #daily_std =  pylab.std(all_cities_yearly_temp, axis=1)
        #all_cities_avg_std = daily_std.mean()
        #annual_std_dev.append(all_cities_avg_std)
        annual_std_dev.append(std_dev)
    annual_std_dev = pylab.array(annual_std_dev)  # converting list to an array.
    return annual_std_dev


def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model’s estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points. 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    for model in models:
        p = pylab.poly1d(model)
        y_estim = p(x)
        rmse_model = rmse(y, y_estim)
        deg_model = len(model) - 1

        pylab.figure()
        pylab.scatter(x, y, color='blue')
        pylab.plot(x,y_estim, color='red')

        pylab.xlabel('Years')
        pylab.ylabel('degrees Celsius')

        pylab.title('Modelling temperature change over the years: Testing data' + "\n" +
                        "RMSE = " + str(rmse_model) + "/n" + "Degree of model = " + str(deg_model))
        pylab.show()

if __name__ == '__main__':

    pass 

    # Part A.4
    # Modelling temperatures on Jan 10th in New York, over the years 1961-2009
    climate = Climate("data.csv")
    y = []
    x = []
    for year in TRAINING_INTERVAL:
        y.append(climate.get_daily_temp('NEW YORK', 1, 10, year))
        x.append(year)

    models = generate_models(pylab.array(x), pylab.array(y), [1])
    evaluate_models_on_training(pylab.array(x), pylab.array(y), models)

    # Modelling average yearly temperatures in New York over the years 1961-2009
    climate = Climate("data.csv")
    y = []
    x = []
    for year in TRAINING_INTERVAL:
        yearly_temp = climate.get_yearly_temp('NEW YORK', year)
        yearly_temp_avg = sum(yearly_temp)/len(yearly_temp)
        y.append(yearly_temp_avg)
        x.append(year)

    models = generate_models(pylab.array(x), pylab.array(y), [1])
    evaluate_models_on_training(pylab.array(x), pylab.array(y), models)

    # Part B
    # Modelling national average yearly temperatures over the years 1961-2009
    climate = Climate("data.csv")
    x = [year for year in TRAINING_INTERVAL]
    y = gen_cities_avg(climate, CITIES, TRAINING_INTERVAL)

    models = generate_models(pylab.array(x), pylab.array(y), [1])
    evaluate_models_on_training(pylab.array(x), pylab.array(y), models)

    # Part C
    # Modelling moving average over 5 years of national yearly temperatures data for the years 1961-2009
    # The moving average allows us to emphasize the general/global trend over local/yearly fluctuation.
    climate = Climate("data.csv")
    x = [year for year in TRAINING_INTERVAL]
    y = gen_cities_avg(climate, CITIES, TRAINING_INTERVAL)

    mov_avg = moving_average(y, window_length=5)
    models = generate_models(pylab.array(x), mov_avg, [1])
    evaluate_models_on_training(pylab.array(x), mov_avg, models)

    # Part D.2
    # Modelling moving average over 5 years of national yearly temperatures data for the years 1961-2009
    climate = Climate("data.csv")
    x_train = [year for year in TRAINING_INTERVAL]
    y_train = gen_cities_avg(climate, CITIES, TRAINING_INTERVAL)

    mov_avg_train = moving_average(y_train, window_length=5)
    models = generate_models(pylab.array(x_train), mov_avg_train, [1, 2, 20])
    evaluate_models_on_training(pylab.array(x_train), mov_avg_train, models)

    # Predicting moving average over 5 years of national yearly temperatures data for the years 2010-2015
    x_test = [year for year in TESTING_INTERVAL]
    y_test = gen_cities_avg(climate, CITIES, TESTING_INTERVAL)
    mov_avg_test = moving_average(y_test, window_length=5)
    evaluate_models_on_testing(pylab.array(x_test), mov_avg_test, models)

    # Part E
    # In addition to raising temperature, global warming also makes temperatures more extreme
    # (e.g., very hot or very cold). We surmise that we can model this effect by measuring the
    # standard deviation in our data. A small standard deviation would suggest that the data
    # is very close together around the mean. A larger standard deviation, however, would suggest
    # that the data varies a lot (i.e., more extreme weather). Therefore, we expect that over time,
    # the standard deviation should increase.
    climate = Climate("data.csv")
    x_train = [year for year in TRAINING_INTERVAL]
    y_train_dev = gen_std_devs(climate, CITIES, TRAINING_INTERVAL)
    mov_avg_dev_train = moving_average(y_train_dev, window_length=5)
    models = generate_models(pylab.array(x_train), mov_avg_dev_train, [1])
    evaluate_models_on_training(pylab.array(x_train), mov_avg_dev_train, models)