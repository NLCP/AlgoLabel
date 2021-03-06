from datetime import datetime
from pprint import pprint as pp


def save_result_log(args, result, metrics, ds=None):

    """
    Utility function for printing results to a log file.
    :param args:
    :param result:
    :param metrics:
    :param ds:
    :return:
    """

    now       = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    model     = args["model"]

    with open("./logs/{}_result_log".format(model), "a") as f:

        f.write("[{}]:\n".format(dt_string))

        if not ds:
            ds = "entire"

        f.write("Experiment on the {} dataset:\n".format(ds))

        params = args["models"][model]

        pp(params, indent=4, stream=f)
        pp(params["classifiers"], indent=4, stream=f)
        pp(params["encoders"], indent=4, stream=f)
        f.write("--------------------\n")
        f.write(str(result))
        f.write("--------------------\n")
        for key in metrics:
            f.write("\t{}:\t{}\n".format(key, metrics[key]))
        f.write("============================================================================\n")
        f.flush()
