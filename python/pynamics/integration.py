import pynamics
import numpy
import logging
logger = logging.getLogger('pynamics.integration')

def integrate(*args,**kwargs):
    if pynamics.integrator==0:
        return integrate_odeint(*args,**kwargs)
    elif pynamics.integrator==1:
        newargs = args[0],args[2][0],args[1],args[2][-1]
        return integrate_rk(*newargs ,**kwargs)



def integrate_odeint(*arguments,**keyword_arguments):
    import scipy.integrate
    
    logger.info('beginning integration')

    result = scipy.integrate.odeint(*arguments,**keyword_arguments)

    # ode_func = arguments[0]  # The ODE function
    # y0 = arguments[1]  # Initial conditions as a NumPy array
    # t_eval = arguments[2]  # Time points
    # from scipy.integrate import solve_ivp
    # sol = solve_ivp(
    #     fun=lambda t, y: ode_func(t, y),  # Adapt ode_func for solve_ivp
    #     t_span=(t_eval[0], t_eval[-1]),  # Time span (first and last time points)
    #     y0=y0,
    #     t_eval=t_eval,  # Evaluate at your specified time points
    #     method='RK45'  # Choose a suitable method (RK45, BDF, etc.)
    # )

    logger.info('finished integration')
    return result

def integrate_rk(*arguments,**keyword_arguments):
    import scipy.integrate
    
    logger.info('beginning integration')
    try:
        result = scipy.integrate.RK45(*arguments,**keyword_arguments)
        y = [result.y]
        while True:
            result.step()
            y.append(result.y)
    except RuntimeError:
        pass
    logger.info('finished integration')
    return y
