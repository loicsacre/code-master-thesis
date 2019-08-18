import datetime
import time


def duration_str(start, end):
    return "{}".format(datetime.timedelta(seconds=end - start))


class Chrono(object):
    """iterator must have a length"""
    def __init__(self, iterator, label, update_rate=0.1,
                 eta_decay_rate=.9, interactive=False):
        self.iterator = iterator
        self.label = label
        self.update_rate = update_rate
        self.decay_rate = eta_decay_rate
        self.interactive = interactive

    def __iter__(self):
        length = len(self.iterator)
        print_it = False
        last_print_iteration = 0
        start = time.time()
        end = start
        average_speed = None
        for i, x in enumerate(self.iterator):
            yield x

            i += 1
            # Should we print ?
            if (i - last_print_iteration) / length >= self.update_rate:
                print_it = True

            # Do print if needed
            if print_it:
                # Compute estimated time of arrival
                now = time.time()
                duration = now - end
                current_speed = (i - last_print_iteration) / duration
                if average_speed is None:
                    average_speed = current_speed
                average_speed = self.decay_rate * average_speed \
                                + (1 - self.decay_rate) * current_speed
                eta = (length - i) / average_speed

                # Print info
                self.print(i, length, now - start, eta)

                # Reset everything
                last_print_iteration = i
                print_it = False
                end = now

        end = time.time()
        if self.interactive:
            print()
        print("Duration ({}):".format(self.label), duration_str(start, end))

    def print(self, iteration, length, elapsed, eta):
        mask = "{{}}    {:<20} Elapsed {{:>10}} | ETA {{:<13}}" \
               "".format("[{}/{} ({:.0f}%)]".format(iteration,
                                                    length,
                                                    iteration / length * 100))
        line = mask.format(self.label,
                           str(datetime.timedelta(seconds=int(elapsed))),
                           str(datetime.timedelta(seconds=int(eta))))
        if self.interactive:
            print("\r", "{:<80}".format(line), end="", sep="")
        else:
            print("{:<80}".format(line))
