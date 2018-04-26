"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([init_x, init_y, 0., 0.])  # state

        # initialize covariance matrix
        self.covariance = np.eye(4)

        # initialize state transition matrix
        self.d_t = np.eye(4)
        self.d_t[0, 2] = 1  # assuming delta_t = 1 add delta_t updates to d_t
        self.d_t[1, 3] = 1

        # initialize measurement matrix
        self.m_t = np.zeros((2, 4))
        self.m_t[0, 0] = 1
        self.m_t[1, 1] = 1

        # define and initialize noise matrices
        self.process_noise = Q
        self.measure_noise = R

    def predict(self):
        self.state = np.dot(self.d_t, self.state)  #+ self.process_noise
        covar_d_t = np.dot(self.d_t, self.covariance)
        self.covariance = np.dot(covar_d_t, self.d_t.T) + self.process_noise

    def correct(self, meas_x, meas_y):
        # first compute kalman gain
        m_t_covar = np.dot(self.m_t, self.covariance)
        norm_factor = np.dot(m_t_covar, self.m_t.T) + self.measure_noise
        noise_inv = np.linalg.inv(norm_factor)
        k_t = np.dot(np.dot(self.covariance, self.m_t.T), noise_inv)

        # update state and covariance
        residual = np.array([meas_x, meas_y]) - np.dot(self.m_t, self.state)
        self.state = self.state + np.dot(k_t, residual)

        self.covariance = np.dot((np.eye(4) - np.dot(k_t, self.m_t)),
                                 self.covariance)

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_mse (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        # convert template to grayscale for comparison in the PF
        self.template = 0.12 * template[:, :, 0] + \
                        0.58 * template[:, :, 1] + \
                        0.3 * template[:, :, 2]
        self.template = self.template.astype(np.uint8)

        self.mid_x = self.template_rect['w'] // 2
        self.mid_y = self.template_rect['h'] // 2

        self.frame = frame

        # initialize distribution of particles as a normal distribution
        # around the initial template location with large sigma to allow for
        # an initial large dispersion
        means = [self.template_rect['x'] + self.mid_x,
                 self.template_rect['y'] + self.mid_y]
        cov = [[100, 0], [0, 100]]

        self.particles = np.random.multivariate_normal(means, cov,
                                               self.num_particles).astype(int)

        self.weights = np.ones((self.num_particles, 1)) / self.num_particles  #

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """

        # determine residual between template and frame cutout at each pixel
        diff = template - frame_cutout

        # calculate mean squared error for all pixels in window cutout
        mse = (np.sum(np.square(diff))) / (template.shape[0] *
                                          template.shape[1])

        # convert mse to a similarity measure between template and frame cutout
        error_measure = np.exp(-mse / (2 * self.sigma_exp ** 2))

        return error_measure

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """

        # flatten weights array into a 1D array
        weights = np.ravel(self.get_weights())

        # randomly choose indices in particles array by index based on weights
        particle_idx = range(self.num_particles)
        new_particle_idx = np.random.choice(particle_idx,
                                            size=self.num_particles,
                                            p=weights)

        # assign particles to new array by using chosen indices
        new_particles = np.array(self.get_particles()[new_particle_idx])

        return new_particles

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        This method is overloaded in the MDParticleFilter class.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """

        # convert the current frame to grayscale to be compared to the
        # grayscale template
        frame = 0.12 * frame[:, :, 0] + \
                0.58 * frame[:, :, 1] + \
                0.3 * frame[:, :, 2]
        self.frame = frame

        # convert template image to grayscale using luma value
        # (0.3, 0.58, 0.12) and trim last row/col if odd shape
        template_gray = self.template.astype(float)

        if template_gray.shape[0] % 2 == 1:
            template_gray = template_gray[:-1, :]
        if template_gray.shape[1] % 2 == 1:
            template_gray = template_gray[:, :-1]

        # initialize factor for normalizing all weights
        normalization_factor = 0.0

        # iterate over each resampled particle to adjust weights
        for i in range(self.num_particles):
            resize_true = False

            v, u = self.particles[i]

            # apply the dynamics update to u and v independently
            delta_u = np.random.normal(0, self.sigma_dyn)
            delta_v = np.random.normal(0, self.sigma_dyn)

            new_u = u + delta_u
            new_v = v + delta_v

            # re-sample dynamics update if it leads to movement outside frame
            while (new_u < 0) or (new_u >= self.frame.shape[0] - 1):
                delta_u = np.random.normal(0, self.sigma_dyn)
                new_u = u + delta_u
            while (new_v < 0) or (new_v >= self.frame.shape[1] - 1):
                delta_v = np.random.normal(0, self.sigma_dyn)
                new_v = v + delta_v

            u = int(round(new_u))
            v = int(round(new_v))

            # update particle list to include new particle position
            self.particles[i] = [v, u]

            # determine bounds for a comparison window from frame
            lower_u, upper_u = u - self.mid_y, u + self.mid_y
            lower_v, upper_v = v - self.mid_x, v + self.mid_x

            # check to see if we need to trim the window if particle close to
            #  edge of frame
            check_top, check_bot = self.mid_y, self.mid_y
            check_left, check_right = self.mid_x, self.mid_x

            if lower_u < 0:
                check_top = u - 0
                lower_u = 0
                resize_true = True
            if lower_v < 0:
                check_left = v - 0
                lower_v = 0
                resize_true = True
            if upper_u > frame.shape[0]:
                check_bot = frame.shape[0] - u
                upper_u = frame.shape[0]
                resize_true = True
            if upper_v > frame.shape[1]:
                check_right = frame.shape[1] - v
                upper_v = frame.shape[1]
                resize_true = True

            # resize template to match shrunken window in case we have
            # particles near edge of frame
            if resize_true:
                template_gray_comp = template_gray[self.mid_y - check_top:
                self.mid_y + check_bot,
                                     self.mid_x - check_left:
                                     self.mid_x + check_right]
                frame_cutout = frame[u - check_top:
                u + check_bot, v - check_left: v + check_right].astype(float)
            else:
                template_gray_comp = template_gray
                frame_cutout = frame[lower_u: upper_u, lower_v:
                upper_v].astype(float)

            # update weight using 'sensor model' similarity measure
            w_i = self.get_error_metric(template_gray_comp, frame_cutout)
            self.weights[i] = w_i

            normalization_factor += w_i

        # normalize weights
        self.weights = self.weights / normalization_factor

        # resample particles
        self.particles = self.resample_particles()

        # update template if appearance of template is changing
        if isinstance(self, AppearanceModelPF):
            self.template = self.update_template(self.template)

    def get_center_mass(self):
        """ Calculates the center of mass for the current set of particles. 
        
        Center of mass of particles is calculated by finding the mean of ((x, 
        y) * weight) across all current particles.

        This method is overloaded in the MDParticleFilter class.

        Returns:
            v: x-coordinate of center of mass as a float
            u: y-coordinate of center of mass as a float
        """

        all_particles = self.get_particles()
        weights = self.get_weights()

        # initialize weighted means and average distance to these means
        x_weighted_mean = 0
        y_weighted_mean = 0

        # calculate the weighted mean of all particles
        for i in range(self.num_particles):
            v, u = all_particles[i]

            x_weighted_mean += v * weights.item(i)
            y_weighted_mean += u * weights.item(i)

        return x_weighted_mean, y_weighted_mean

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        This method is overloaded in the MDParticleFilter class.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        # initialize avg dist to predicted location
        dist = 0
        particles = self.get_particles()

        # calculate the predicted location of the object in the current frame
        # and also get proper rescaling for bounding box if MDParticleFilter
        if isinstance(self, MDParticleFilter):
            v, u, avg_scale = self.get_center_mass()
            new_template = cv2.resize(self.template, dsize=None,
                                      fx=avg_scale, fy=avg_scale,
                                      interpolation=cv2.INTER_AREA)

            mid_x = new_template.shape[1] // 2
            mid_y = new_template.shape[0] // 2

        elif isinstance(self, MultiParticleFilter):
            v, u, _, _ = self.get_center_mass()

            mid_x, mid_y = self.mid_x, self.mid_y
        else:
            v, u = self.get_center_mass()

            mid_x, mid_y = self.mid_x, self.mid_y

        # define top left and bot-right points for window rectangle
        pt_1 = (int(round(v)) - mid_x,
                int(round(u)) - mid_y)
        pt_2 = (int(round(v)) + mid_x,
                int(round(u)) + mid_y)
        cv2.rectangle(frame_in, pt_1, pt_2, color=(255, 255, 0), thickness=2)

        # determine the average distance of all particles to mean (x, y) and
        # also draw the current position of each particle
        for i in range(self.num_particles):
            if isinstance(self, MDParticleFilter):
                x, y, _ = particles[i]
            elif isinstance(self, MultiParticleFilter):
                x, y, _, _ = particles[i]
            else:
                x, y = particles[i]

            # draw the particle
            color = (0, 0, 255)  # set the color for each particle
            cv2.circle(frame_in, center=(int(round(x)), int(round(y))),
                       radius=2,
                       color=color,
                       thickness=1)

            x_dev = x - v
            y_dev = y - u

            dist += np.sqrt(x_dev ** 2 + y_dev ** 2) * self.weights.item(i)

        # draw circle around mean (x, y) with radius equal to avg dist
        cv2.circle(frame_in, center=(int(round(v)), int(round(u))),
                   radius=int(round(dist)),
                   color=(0, 0, 0),
                   thickness=2)


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.

        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)
        # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default
        # value so that your test doesn't fail the autograder because of an
        # unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def update_template(self, template_img):
        """ This function updates the template using an Infinite Impulse
        Response filter by updating the template as a weighted sum of the
        current template and the frame cutout from the best curent location.
        This function is called in the base class process function, using an
        isinstance check to determine if object is an 'ApperanceModelPF' object.

        This method is overloaded in the MDParticleFilter class.

        Args:
            template_img (numpy.array): template image as grayscale float.
        Returns:
            updated_template (numpy.array): updated template as a weighted sum
                                            of recent best frame cutout and
                                            current template.
        """

        # intiialize
        alpha = self.alpha
        beta = (1 - alpha)
        frame_in = self.frame

        # convert frame cutout to grayscale using luma value (0.3, 0.58, 0.12)
        if template_img.shape[0] % 2 == 1:
            template_img = template_img[:-1, :]
        if template_img.shape[1] % 2 == 1:
            template_img = template_img[:, :-1]

        template_img = template_img.astype(float)
        output_template = template_img.copy()

        # determine the mean predicted location
        v, u = self.get_center_mass()
        v, u = int(round(v)), int(round(u))

        # determine bounds for a comparison window from frame
        lower_u, upper_u = u - self.mid_y, u + self.mid_y
        lower_v, upper_v = v - self.mid_x, v + self.mid_x

        # if frame bounds are possible then update template
        if lower_u > 0 and upper_u < self.frame.shape[0] \
                and lower_v > 0 and upper_v < self.frame.shape[1]:
            frame_cutout = frame_in[lower_u: upper_u,
                           lower_v: upper_v].astype(float)

            output_template = cv2.addWeighted(frame_cutout, alpha,
                                              template_img, beta, 0)

        return output_template.astype(np.uint8)


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)
        # call base class constructor

        # define sigma for varying rescale value
        self.sigma_scale = kwargs.get('sigma_scale', 0.04)

        # define tracking variables to be updated during tracking
        self.current_position = (0, 0)
        self.all_scales = [1.0]
        self.all_displacements = []
        self.all_avg_weights = []
        self.curr_scale = 1.0

        # define threshold for determining when object is occluded
        self.w_threshold = kwargs.get('w_threshold', 5E-4)
        self.initial_scales = kwargs.get('initial_scales', [0.5, 1.0])

        # initialize distribution of particles [x, y, s] as a normal
        # distribution around the initial template location for (x, y)
        # and a uniformly random scale s determined by self.initial_scales
        means = [self.template_rect['x'] + self.mid_x,
                 self.template_rect['y'] + self.mid_y]
        cov = [[150, 0], [0, 150]]

        mid_scale = np.mean(self.initial_scales)
        range = (self.initial_scales[1] - self.initial_scales[0]) // 2
        scales = range * np.random.rand(self.num_particles, 1) + mid_scale

        self.particles = np.random.multivariate_normal(means, cov,
                                               self.num_particles).astype(int)
        self.particles = np.hstack((self.particles, scales))

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.
        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.
        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].
        Returns:
            None.
        """

        # initialize factor for normalizing all weights
        normalization_factor = 0.0

        # convert current frame to grayscale
        frame = 0.12 * frame[:, :, 0] + \
                0.58 * frame[:, :, 1] + \
                0.3 * frame[:, :, 2]
        self.frame = frame

        # iterate over each resampled particle to adjust weights
        for i in range(self.num_particles):
            resize_true = False

            v, u, scale = self.particles[i]

            delta_scale = np.random.randn(1) * self.sigma_scale + 1
            new_scale = scale * delta_scale
            if new_scale > 1.0:
                new_scale = 1.0

            template = cv2.resize(self.template, dsize=None, fx=new_scale,
                                  fy=new_scale, interpolation=cv2.INTER_AREA)

            mid_x = template.shape[1] // 2
            mid_y = template.shape[0] // 2

            # convert template image to grayscale using luma value
            # (0.3, 0.58, 0.12) and trim last row/col if odd shape
            template_gray = template.astype(float)

            if template_gray.shape[0] % 2 == 1:
                template_gray = template_gray[:-1, :]
            if template_gray.shape[1] % 2 == 1:
                template_gray = template_gray[:, :-1]

            # apply the dynamics update to u, v, and scale independently
            delta_u = np.random.normal(0, self.sigma_dyn)
            delta_v = np.random.normal(0, self.sigma_dyn)

            new_u = u + delta_u
            new_v = v + delta_v

            # re-sample dynamics update if it leads to movement outside frame
            while (new_u < 0) or (new_u >= self.frame.shape[0] - 1):
                delta_u = np.random.normal(0, self.sigma_dyn)
                new_u = u + delta_u
            while (new_v < 0) or (new_v >= self.frame.shape[1] - 1):
                delta_v = np.random.normal(0, self.sigma_dyn)
                new_v = v + delta_v

            u = int(round(new_u))
            v = int(round(new_v))

            # update particle list to include new particle position
            self.particles[i] = [v, u, new_scale]

            # determine bounds for a comparison window from frame
            lower_u, upper_u = u - mid_y, u + mid_y
            lower_v, upper_v = v - mid_x, v + mid_x

            # check to see if we need to trim the window if particle close to
            #  edge of frame
            check_top, check_bot = mid_y, mid_y
            check_left, check_right = mid_x, mid_x

            if lower_u < 0:
                check_top = u - 0
                lower_u = 0
                resize_true = True
            if lower_v < 0:
                check_left = v - 0
                lower_v = 0
                resize_true = True
            if upper_u > frame.shape[0]:
                check_bot = frame.shape[0] - u
                upper_u = frame.shape[0]
                resize_true = True
            if upper_v > frame.shape[1]:
                check_right = frame.shape[1] - v
                upper_v = frame.shape[1]
                resize_true = True

            # resize template to match shrunken window in case we have
            # particles near edge of frame
            if resize_true:
                template_gray_comp = template_gray[mid_y - check_top:
                mid_y + check_bot,
                                     mid_x - check_left:
                                     mid_x + check_right]
                frame_cutout = frame[u - check_top:
                u + check_bot,
                               v - check_left:
                               v + check_right].astype(float)
            else:
                template_gray_comp = template_gray
                frame_cutout = frame[lower_u: upper_u, lower_v:
                upper_v].astype(float)

            # update weight using 'sensor model' similarity measure
            w_i = self.get_error_metric(template_gray_comp, frame_cutout)
            self.weights[i] = w_i

            normalization_factor += w_i

        # update tracking vals, weights, and resample; also check for occlusion
        self.update_tracking_vals(normalization_factor)

    def get_center_mass(self):
        """ Calculates the center of mass for the current set of particles. 

        Center of mass of particles is calculated by finding the mean of ((x,
        y, s) * weight) across all current particles. For the
        MDParticleFilter class, this method now also calculates mean scale of
        particles after weight update is complete so as to only include best /
        near-best scales.

        Returns:
            v: x-coordinate of center of mass as a float
            u: y-coordinate of center of mass as a float
            scale: template rescaling proportion as a float
        """

        particles = self.get_particles()
        weights = self.get_weights()

        # initialize weighted means and average distance to these means
        x_weighted_mean = 0
        y_weighted_mean = 0
        scale = 0.0

        # calc the weighted mean
        for i in range(self.num_particles):
            v, u, new_scale = particles[i]

            x_weighted_mean += v * weights.item(i)
            y_weighted_mean += u * weights.item(i)
            scale += new_scale * weights.item(i)

        return x_weighted_mean, y_weighted_mean, scale

    def update_template(self, template_img):
        """This function updates the template using an Infinite Impulse Response
        filter by updating the template as a weighted sum of the current
        template and the frame cutout from the best curent location.

        This function is called in the base class process function, using an
        isinstance check to determine if object is an 'ApperanceModelPF' obj.

        Args:
            template_img (numpy.array): template image as grayscale float.

        Returns:
            updated_template (numpy.array): updated template as a weighted sum
                                            of recent best frame cutout and
                                            current template.
        """

        # initialize
        alpha = self.alpha
        beta = 1 - alpha

        # get a resized version of the template at the avg best scale from
        # previous frame to determine size of current frame
        best_template = cv2.resize(template_img, dsize=None,
                                   fx=self.all_scales[-1],
                              fy=self.all_scales[-1],
                              interpolation=cv2.INTER_AREA)

        # convert frame cutout to grayscale using luma value (0.3, 0.58, 0.12)
        if best_template.shape[0] % 2 == 1:
            best_template = best_template[:-1, :]
        if best_template.shape[1] % 2 == 1:
            best_template = best_template[:, :-1]

        mid_x = best_template.shape[1] // 2
        mid_y = best_template.shape[0] // 2

        template = best_template.astype(float)

        # plot each particle as a white dot and calc the weighted mean (x, y)
        v, u = self.current_position
        u, v = int(round(u)), int(round(v))

        # determine bounds for a comparison window from frame
        lower_u, upper_u = u - mid_y, u + mid_y
        lower_v, upper_v = v - mid_x, v + mid_x

        # if frame bounds are possible then update template using scaled up
        # version of best frame cutout
        if lower_u > 0 and upper_u < self.frame.shape[0] \
                and lower_v > 0 and upper_v < self.frame.shape[1]:
            frame_cutout = self.frame[lower_u: upper_u,
                           lower_v: upper_v].astype(float)

            # scale up frame cutout to match original template size
            best_frame = cv2.resize(frame_cutout,
                                         dsize=(template_img.shape[1],
                                                template_img.shape[0]),
                                 interpolation=cv2.INTER_CUBIC).astype(
                np.uint8)
            output_template = cv2.addWeighted(best_frame, alpha,
                                              template_img, beta, 0)

        else:
            output_template = template

        return output_template

    def update_tracking_vals(self, normalization_factor):
        """ Update average displacement and average template error values.

        The top 5% of un-normalized weights are averaged for the current
        frame, with a running window of these averages used to threshold
        whenever a template is found (along with a multiplicative factor
        self.w_threshold.

        If template does not meet threshold, self.estimate_next_frame is
        called to handle what is assumed to be occlusion. See docstring in
        that method for more information. If template does meet threshold,
        update weights, resample, and update tracking variables for the
        current best scale and the current location (x, y).

        Args:
            normalization_factor: sum of all weights to normalize weights as
                                a float value
        """

        # calculate the running average of top weights
        avg_weight = np.mean(self.weights)
        self.all_avg_weights.append(avg_weight)
        top_5_percent = np.ceil(0.95 * len(self.all_avg_weights)).astype(int)
        running_avg = np.mean(self.all_avg_weights[top_5_percent - 1:])

        # determine whether best template match in current frame is within
        # threshold
        if max(self.weights) < self.w_threshold * running_avg:

            self.estimate_next_frame()

        else:
            # normalize weights
            self.weights = self.weights / normalization_factor

            # resample particles
            self.particles = self.resample_particles()

            # get best location and scale of object
            v, u, avg_scale = self.get_center_mass()

            self.all_scales.append(avg_scale)

            # if this is the first frame, initialize current position else
            # update position and displacement
            if self.current_position == (0, 0):
                self.current_position = (v, u)
            else:
                last_x, last_y = self.current_position

                self.current_position = (v, u)

                # update displacements
                diff_x, diff_y = self.current_position[0] - last_x, \
                                 self.current_position[1] - last_y
                self.all_displacements.append((diff_x, diff_y))

            # update template to incorporate appearance changes
            self.template = self.update_template(self.template)

    def dist_error(self, test_x, test_y, new_x, new_y):
        """ Returns an error metric based simply on how distant particles are
        from a predicted location using simple euclidean distance.

        This metric is simply used to keep particles close to where the
        object is predicted to be, simulating "tracking" during
        occlusion.

        self.sigma_exp is passed in through experiment.py, and default value
        is set to 0.04, determined through experimentation.

        Args:
            test_x: x-coordinate of current particle as a float.
            test_y: y-coordinate of current particle as a float.
            new_x: x-coordinate of predicted location as a float.
            new_y: y-coordinate of predicted location as a float.
        Returns:
            error_measure: similarity value returned as a float.
        """

        # determine residual between template and frame cutout at each pixel
        diff_x, diff_y = test_x - new_x, test_y - new_y

        # calculate euclidean distance between locations
        dist = np.sqrt(diff_x ** 2 + diff_y ** 2)

        # convert dist to a similarity measure between particle and predicted
        #  loc
        error_measure = np.exp(-dist / (2 * self.sigma_exp ** 2))

        return error_measure

    def estimate_next_frame(self):
        """ Estimate the next frame if no reasonable template is found
        in the image, hence simulating tracking during occlusion.

        This method handles occlusion by updating new location of the object
        in the current frame to be a new predicted location, determined by
        extrapolating from previously tracked average displacements.

        The ideal scale for the template also continues to scale down each
        frame by 0.99 to approximate the size of the object getting smaller
        by a small amount after occlusion.

        Particles are still resampled during occlusion, and this is done so
        as to allow for a varied distribution around the predicted point to
        hopefully pick up the object after occlusion with the right (x, y, s).

        """

        # initialize
        normalization_factor = 0.0

        # do not update top average weights
        self.all_avg_weights[-1] = self.all_avg_weights[-2]

        # use previous scale to determine next scale
        new_scale = self.all_scales[-1] * 0.99

        # add this new scale to the tracking list of rescale values
        self.all_scales.append(new_scale)

        # update new location if template has been previously successfully
        # tracked
        try:
            avg_displacement = np.mean(self.all_displacements[1:], axis=0)

            new_x, new_y = self.current_position[0] + avg_displacement[0],\
                           self.current_position[1] - 2 * avg_displacement[1]
        except:
            new_x, new_y = self.current_position

        self.current_position = (new_x, new_y)

        # re-distribute particles about the predicted new location using the
        # same gaussian as dynamics model, and randomize scales as gaussian
        # about previous best scale
        means = [int(round(new_x)), int(round(new_y))]
        cov = [[self.sigma_dyn, 0], [0, self.sigma_dyn]]
        scales = np.random.randn(self.num_particles, 1) * self.sigma_scale + \
                 new_scale

        self.particles = np.random.multivariate_normal(means, cov,
                                                       self.num_particles).astype(
            int)
        self.particles = np.hstack((self.particles, scales))

        # update weights based on distance to predicted location
        for i in range(self.num_particles):
            test_x, test_y, _ = self.particles[i]

            w_i = self.dist_error(test_x, test_y, new_x, new_y)

            normalization_factor += w_i

            self.weights[i] = w_i

        # normalize weights
        self.weights = self.weights / normalization_factor

        # resample particles
        self.particles = self.resample_particles()


class MultiParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics.
    
    This class extends the Appearance model to incorporate more dynamics, 
    similar to the MDParticleFilter, but instead of incorporating scale as 
    an additional dynamic, this class also incorporates velocity in both x 
    and y directions, and also uses a more robust similarity measure.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MultiParticleFilter, self).__init__(frame, template, **kwargs)
        # call base class constructor

        # define parameters for histograms used in new similarity function
        self.hist_bins = kwargs.get('hist_bins', 5)
        self.ratio_template_hist = kwargs.get('ratio_template_hist', 5)

        # define mean initial velocities for particles
        self.initial_vel_x = kwargs.get('initial_vel_x', 0)
        self.initial_vel_y = kwargs.get('initial_vel_y', 0)

        # define the multiplicative threshold value for detecting occlusion
        self.w_threshold = kwargs.get('w_threshold', 1E-4)

        # store full color original template for histogram comparison
        self.orig_template = template

        # define tracking variables to be updated during tracking
        self.all_avg_weights = []
        self.current_position = (0, 0)
        self.all_displacements = []

        # create normal distributions around the template location and the
        # around the defined initial velocity estimate
        means = [self.template_rect['x'] + self.mid_x,
                 self.template_rect['y'] + self.mid_y,
                 self.initial_vel_x,
                 self.initial_vel_y]
        cov = [[150, 0, 0, 0], [0, 150, 0, 0], [0, 0, 150, 0], [0, 0, 0, 150]]

        self.particles = np.random.multivariate_normal(means, cov,
                                                       self.num_particles).astype(
            int)

    def get_error_metric_new(self, template, frame_cutout, template_col,
                         frame_col):
        """Returns the error metric used based on the similarity measure.
        
        The similarity measure has been expanded in the MultiParticleFilter 
        to also use a comparison of color histograms between the original 
        color template and color image of the current frame. 
        
        A weighting factor self.ratio_template_hist defines the weighting 
        between template matching and histogram comparison in contributing 
        to the overall similarity measure between template and frame_cutout.

        Args:
            template (numpy.array): copy of template in grayscale.
            frame_cutout (numpy.array): copy of frame cutout around particle
                            location to be compared to template in grayscale.
            template_col (numpy.array): copy of template in colour.
            frame_col (numpy.array): copy of frame cutout around particle
                            location to be compared to template in grayscale
                            using color histograms.
        Returns:
            float: similarity value.
        """

        # create color histograms for template and current frame cutout
        template_hist = cv2.calcHist([template_col.astype('float32')],
                                     [2],
                                     mask=np.ones(template_col.shape[
                                                  :2]).astype('uint8'),
                                     histSize=[self.hist_bins],
                                     ranges=[0, 255])

        frame_hist = cv2.calcHist([frame_col.astype('float32')],
                                  [2],
                                  mask=np.ones(frame_col.shape[:2]).astype(
                                      'uint8'),
                                  histSize=[self.hist_bins],
                                  ranges=[0, 255])

        # calculate the mean squared error for all bins between histograms
        hist_diff = template_hist - frame_hist
        hist_mse = (np.sum(hist_diff ** 2)) / (template.shape[0] *
                                               template.shape[1])

        # determine residual between template and frame cutout at each pixel
        diff = template - frame_cutout

        # calculate mean squared error for all pixels in window cutout
        mse = (np.sum(diff ** 2)) / (template.shape[0] *
                                     template.shape[1])

        # scale template match mean squared error by weight controlling
        # importance balance between template matching and histogram comparison
        mse *= self.ratio_template_hist

        # convert mse to a similarity measure between template and frame cutout
        error_measure = np.exp(-(mse + hist_mse) / (2 * self.sigma_exp ** 2))

        return error_measure

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.
        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.
        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.
        
        This method has been expanded from AppearanceParticleFilter to also 
        create color template and frame cutouts using the new similarity 
        measure used in the MultiParticleFilter class. Also this method updates
        the particles storing the state (x, y, vx, vy) updating based on the 
        new dynamics model using vx and vy to update x and y.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].
        Returns:
            None.
        """

        # initialize factor for normalizing all weights
        normalization_factor = 0.0

        template_col = self.orig_template.astype('float32')
        frame_col = frame.astype('float32')

        # store color version of frame in self.orig_frame
        self.orig_frame = frame

        # convert current frame to grayscale
        frame = 0.12 * frame[:, :, 0] + \
                0.58 * frame[:, :, 1] + \
                0.3 * frame[:, :, 2]
        self.frame = frame

        # iterate over each resampled particle to adjust weights
        for i in range(self.num_particles):
            resize_true = False

            v, u, vel_x, vel_y = self.particles[i]

            mid_x = self.template.shape[1] // 2
            mid_y = self.template.shape[0] // 2

            # convert template image to grayscale using luma value
            # (0.3, 0.58, 0.12) and trim last row/col if odd shape
            template_gray = self.template.astype(float)

            if template_gray.shape[0] % 2 == 1:
                template_gray = template_gray[:-1, :]
                template_col = template_col[:-1, :, :]
            if template_gray.shape[1] % 2 == 1:
                template_gray = template_gray[:, :-1]
                template_col = template_col[:, :-1, :]

            # apply the dynamics update to u, v, and scale independently
            delta_vel_x = np.random.normal(0, self.sigma_dyn)
            delta_vel_y = np.random.normal(0, self.sigma_dyn)

            new_vel_x = vel_x + delta_vel_x
            new_vel_y = vel_y + delta_vel_y

            new_u = u + new_vel_x
            new_v = v + new_vel_y

            # re-sample dynamics update if it leads to movement outside frame
            while (new_u < 0) or (new_u >= self.frame.shape[0] - 1):
                delta_vel_y = np.random.normal(0, self.sigma_dyn)
                new_vel_y = vel_y + delta_vel_y
                new_u = u + new_vel_y
            while (new_v < 0) or (new_v >= self.frame.shape[1] - 1):
                delta_vel_x = np.random.normal(0, self.sigma_dyn)
                new_vel_x = vel_x + delta_vel_x
                new_v = v + new_vel_x

            u = int(round(new_u))
            v = int(round(new_v))

            # update particle list to include new particle position
            self.particles[i] = [v, u, new_vel_x, new_vel_y]

            # determine bounds for a comparison window from frame
            lower_u, upper_u = u - mid_y, u + mid_y
            lower_v, upper_v = v - mid_x, v + mid_x

            # check to see if we need to trim the window if particle close to
            #  edge of frame
            check_top, check_bot = mid_y, mid_y
            check_left, check_right = mid_x, mid_x

            if lower_u < 0:
                check_top = u - 0
                lower_u = 0
                resize_true = True
            if lower_v < 0:
                check_left = v - 0
                lower_v = 0
                resize_true = True
            if upper_u > frame.shape[0]:
                check_bot = frame.shape[0] - u
                upper_u = frame.shape[0]
                resize_true = True
            if upper_v > frame.shape[1]:
                check_right = frame.shape[1] - v
                upper_v = frame.shape[1]
                resize_true = True

            # resize template to match shrunken window in case we have
            # particles near edge of frame
            if resize_true:
                template_gray_comp = template_gray[mid_y - check_top:
                mid_y + check_bot, mid_x - check_left: mid_x + check_right]
                frame_cutout = frame[u - check_top:
                u + check_bot, v - check_left: v + check_right].astype(float)

                template_col_comp = template_col[mid_y - check_top:
                               mid_y + check_bot, mid_x - check_left:
                               mid_x + check_right].astype(float)
                frame_col_comp = frame_col[u - check_top:
                            u + check_bot, v - check_left: v +
                                                   check_right].astype(float)
            else:
                template_gray_comp = template_gray
                template_col_comp = template_col
                frame_cutout = frame[lower_u: upper_u, lower_v:
                upper_v].astype(float)
                frame_col_comp = frame_col[lower_u: upper_u, lower_v:
                            upper_v].astype(float)

            w_i = self.get_error_metric_new(template_gray_comp, frame_cutout,
                                         template_col_comp, frame_col_comp)

            normalization_factor += w_i
            self.weights[i] = w_i

        # update tracking vals, weights, and resample; also check for occlusion
        self.update_tracking_vals(normalization_factor)

    def get_center_mass(self):
        """ Calculates the center of mass for the current set of particles. 

        Center of mass of particles is calculated by finding the mean of ((x,
        y, vx, vy) * weight) across all current particles. For the
        MultiParticleFilter class, this method now also calculates mean 
        velocities of particles after weight update is complete.

        Returns:
            v: x-coordinate of center of mass as a float
            u: y-coordinate of center of mass as a float
            vx: velocity in the x dimension as a float
            vy: velocity in the y dimension as a float
        """

        particles = self.get_particles()
        weights = self.get_weights()

        # initialize weighted means and average distance to these means
        x_weighted_mean = 0
        y_weighted_mean = 0
        avg_vel_x = 0
        avg_vel_y = 0

        # calc the weighted mean
        for i in range(self.num_particles):
            v, u, vel_x, vel_y = particles[i]

            x_weighted_mean += v * weights.item(i)
            y_weighted_mean += u * weights.item(i)
            avg_vel_x += vel_x * weights.item(i)
            avg_vel_y += vel_y * weights.item(i)

        return x_weighted_mean, y_weighted_mean, avg_vel_x, avg_vel_y

    def update_tracking_vals(self, normalization_factor):
        """ Update average displacement and average template error values.
        
        This method expands upon the approach used in MDParticleFilter to 
        accommodate the new state represented by MultiParticleFilter.

        The top 5% of un-normalized weights are averaged for the current
        frame, with a running window of these averages used to threshold
        whenever a template is found (along with a multiplicative factor
        self.w_threshold.

        If template does not meet threshold, self.estimate_next_frame is
        called to handle what is assumed to be occlusion. See docstring in
        that method for more information. If template does meet threshold,
        update weights, resample, and update tracking variables for the
        current best scale and the current location (x, y).

        """

        threshold = self.w_threshold

        avg_weight = np.mean(self.weights)

        # determine the top 5 percent of weights and use this to calculate a
        #  running average of top weights
        top_5_percent = np.ceil(0.95 * len(self.all_avg_weights)).astype(int)
        # if this if the first frame use the total average from the first frame
        if self.all_avg_weights == []:
            self.all_avg_weights.append(avg_weight)
        running_avg = np.mean(self.all_avg_weights[top_5_percent - 1:])

        # determine the current best similarity measure
        max_weight = np.max(self.weights)

        # if occlusion is detected after tracking the object for at least 5
        # frames
        if max_weight < running_avg * threshold and \
                        len(self.all_avg_weights) > 5:
                self.estimate_next_frame()
        else:

            # if occlusion is not detected then update normalize weights,
            # resample particles and update displacement tracking value

            self.all_avg_weights.append(avg_weight)

            # normalize weights
            self.weights = self.weights / normalization_factor

            # resample particles
            self.particles = self.resample_particles()

            # get best location and scale of object
            v, u, vel_x, vel_y = self.get_center_mass()

            last_x, _ = self.current_position

            self.all_displacements.append(v - last_x)

            self.current_position = (v, u)

            # update template to incorporate appearance changes
            self.template = self.update_template(self.template)

    def update_template(self, template_img):
        """This function updates the template using an Infinite Impulse Response
        filter by updating the template as a weighted sum of the current
        template and the frame cutout from the best curent location.

        This function is called in the base class process function, using an
        isinstance check to determine if object is an 'ApperanceModelPF' obj.

        Args:
            template_img (numpy.array): template image as grayscale float.

        Returns:
            updated_template (numpy.array): updated template as a weighted sum
                                            of recent best frame cutout and
                                            current template.
        """

        # initialize
        alpha = self.alpha
        beta = 1 - alpha

        # convert frame cutout to grayscale using luma value (0.3, 0.58, 0.12)
        if template_img.shape[0] % 2 == 1:
            template_img = template_img[:-1, :]
        if template_img.shape[1] % 2 == 1:
            template_img = template_img[:, :-1]

        mid_x = template_img.shape[1] // 2
        mid_y = template_img.shape[0] // 2

        template = template_img.astype(float)

        # plot each particle as a white dot and calc the weighted mean (x, y)
        v, u = self.current_position
        u, v = int(round(u)), int(round(v))

        # determine bounds for a comparison window from frame
        lower_u, upper_u = u - mid_y, u + mid_y
        lower_v, upper_v = v - mid_x, v + mid_x

        # if frame bounds are possible then update template using scaled up
        # version of best frame cutout
        if lower_u > 0 and upper_u < self.frame.shape[0] \
                and lower_v > 0 and upper_v < self.frame.shape[1]:
            frame_cutout = self.frame[lower_u: upper_u,
                                lower_v: upper_v].astype(float)
            orig_frame_cutout = self.orig_frame[lower_u: upper_u,
                                lower_v: upper_v].astype(float)

            output_template = cv2.addWeighted(frame_cutout, alpha,
                                              template, beta, 0)
            self.orig_template = cv2.addWeighted(orig_frame_cutout, alpha,
                                                 self.orig_template.astype(
                                                     float), beta, 0)

        else:
            output_template = template

        return output_template

    def dist_error(self, test_x, test_y, new_x, new_y):
        """ Returns an error metric based simply on how distant particles are
        from a predicted location using simple euclidean distance.

        This metric is simply used to keep particles close to where the
        object is predicted to be, simulating "tracking" during
        occlusion.

        self.sigma_exp is passed in through experiment.py, and default value
        is set to 0.04, determined through experimentation.

        Args:
            test_x: x-coordinate of current particle as a float.
            test_y: y-coordinate of current particle as a float.
            new_x: x-coordinate of predicted location as a float.
            new_y: y-coordinate of predicted location as a float.
        Returns:
            error_measure: similarity value returned as a float.
        """

        # determine residual between template and frame cutout at each pixel
        diff_x, diff_y = test_x - new_x, test_y - new_y

        # calculate euclidean distance between locations
        dist = np.sqrt(diff_x ** 2 + diff_y ** 2)

        # convert dist to a similarity measure between particle and predicted
        #  loc
        error_measure = np.exp(-dist / (2 * self.sigma_exp ** 2))

        return error_measure

    def estimate_next_frame(self):
        """ Estimate the next frame if no reasonable template is found
        in the image, hence simulating tracking during occlusion.

        This method handles occlusion by updating new location of the object
        in the current frame to be a new predicted location, determined by
        extrapolating from previously tracked average displacements.

        The ideal scale for the template also continues to scale down each
        frame by 0.99 to approximate the size of the object getting smaller
        by a small amount after occlusion.

        Particles are still resampled during occlusion, and this is done so
        as to allow for a varied distribution around the predicted point to
        hopefully pick up the object after occlusion with the right (x, y, s).

        """

        # initialize
        normalization_factor = 0.0

        # update new location if template has been previously successfully
        # tracked
        try:
            avg_displace = np.mean(self.all_displacements[1:])

            new_x = self.current_position[0] + avg_displace
            new_y = self.current_position[1]
        except:
            avg_displace = 0.0
            new_x, new_y = self.current_position

        self.current_position = (new_x, new_y)

        # re-distribute particles about the predicted new location using the
        # same gaussian as dynamics model
        means = [int(round(new_x)), int(round(new_y)), avg_displace, 0]
        cov = [[self.sigma_dyn, 0, 0, 0], [0, self.sigma_dyn, 0, 0],
               [0, 0, self.sigma_dyn, 0], [0, 0, 0, self.sigma_dyn]]

        self.particles = np.random.multivariate_normal(means, cov,
                                                       self.num_particles).astype(
            int)

        # update weights based on distance to predicted location
        for i in range(self.num_particles):
            test_x, test_y, _, _ = self.particles[i]

            w_i = self.dist_error(test_x, test_y, new_x, new_y)

            normalization_factor += w_i

            self.weights[i] = w_i

        # normalize weights
        self.weights = self.weights / normalization_factor

        # resample particles
        self.particles = self.resample_particles()


class FollowParticleFilter(MultiParticleFilter):
    """A variation of particle filter tracker that incorporates more dynamics.

    This class extends the MultiParticleFilter model to incorporate more
    dynamics, similar to the MDParticleFilter, incorporating scale as
    an additional dynamic as well as also incorporating velocity in both x
    and y directions. This PF also uses a more robust similarity measure,
    such as is used in MultiParticleFilter.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(FollowParticleFilter, self).__init__(frame, template, **kwargs)
        # call base class constructor

        # define parameters for histograms used in new similarity function
        self.hist_bins = kwargs.get('hist_bins', 5)
        self.ratio_template_hist = kwargs.get('ratio_template_hist', 5)

        # define mean initial velocities for particles
        self.initial_vel_x = kwargs.get('initial_vel_x', 0)
        self.initial_vel_y = kwargs.get('initial_vel_y', 0)

        # define the multiplicative threshold value for detecting occlusion
        self.w_threshold = kwargs.get('w_threshold', 1E-4)

        # define the sigma for varying scale dynamics
        self.sigma_scale = kwargs.get('sigma_scale', 0.75)

        # define initial range for scales
        self.initial_scales = kwargs.get('initial_scales', [0.5, 2.0])

        # store full color original template for histogram comparison
        self.orig_template = template

        # define tracking variables to be updated during tracking
        self.all_avg_weights = []
        self.current_position = (0, 0)
        self.all_displacements = []
        self.all_scales = [1.0]

        # create normal distributions around the template location and the
        # around the defined initial velocity estimate
        means = [self.template_rect['x'] + self.mid_x,
                 self.template_rect['y'] + self.mid_y,
                 self.initial_vel_x,
                 self.initial_vel_y]
        cov = [[150, 0, 0, 0], [0, 150, 0, 0], [0, 0, 150, 0], [0, 0, 0, 150]]

        self.particles = np.random.multivariate_normal(means, cov,
                                                       self.num_particles).astype(
            int)

        mid_scale = np.mean(self.initial_scales)
        range = (self.initial_scales[1] - self.initial_scales[0]) // 2
        scales = range * np.random.rand(self.num_particles, 1) + mid_scale

        self.particles = np.hstack((self.particles, scales))

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.
        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.
        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        This method has been expanded from AppearanceParticleFilter to also 
        create color template and frame cutouts using the new similarity 
        measure used in the MultiParticleFilter class. Also this method updates
        the particles storing the state (x, y, vx, vy) updating based on the 
        new dynamics model using vx and vy to update x and y.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].
        Returns:
            None.
        """

        # initialize factor for normalizing all weights
        normalization_factor = 0.0

        frame_col = frame.astype('float32')

        # store color version of frame in self.orig_frame
        self.orig_frame = frame

        # get the smallest and largest scales specified by initial scales
        smallest_scale = self.initial_scales[0]
        largest_scale = self.initial_scales[1]

        # convert current frame to grayscale
        frame = 0.12 * frame[:, :, 0] + \
                0.58 * frame[:, :, 1] + \
                0.3 * frame[:, :, 2]
        self.frame = frame

        # iterate over each resampled particle to adjust weights
        for i in range(self.num_particles):
            resize_true = False

            v, u, vel_x, vel_y, scale = self.particles[i]

            mid_scale = np.mean(self.initial_scales)
            delta_scale = np.random.randn() * self.sigma_scale + mid_scale
            new_scale = scale * delta_scale

            new_scale = max(new_scale, smallest_scale)
            new_scale = min(new_scale, largest_scale)

            template = cv2.resize(self.template, dsize=None, fx=new_scale,
                                  fy=new_scale, interpolation=cv2.INTER_CUBIC)
            template_col = cv2.resize(self.orig_template, dsize=None,
                                      fx=new_scale,
                                      fy=new_scale,
                                      interpolation=cv2.INTER_CUBIC)

            mid_x = template.shape[1] // 2
            mid_y = template.shape[0] // 2

            # convert template image to grayscale using luma value
            # (0.3, 0.58, 0.12) and trim last row/col if odd shape
            template_gray = template.astype(float)

            if template_gray.shape[0] % 2 == 1:
                template_gray = template_gray[:-1, :]
                template_col = template_col[:-1, :, :]
            if template_gray.shape[1] % 2 == 1:
                template_gray = template_gray[:, :-1]
                template_col = template_col[:, :-1, :]

            # apply the dynamics update to u, v, and scale independently
            delta_vel_x = np.random.normal(0, self.sigma_dyn)
            delta_vel_y = np.random.normal(0, self.sigma_dyn)

            new_vel_x = vel_x + delta_vel_x
            new_vel_y = vel_y + delta_vel_y

            new_u = u + new_vel_x
            new_v = v + new_vel_y

            # re-sample dynamics update if it leads to movement outside frame
            while (new_u < 0) or (new_u >= self.frame.shape[0] - 1):
                delta_vel_y = np.random.normal(0, self.sigma_dyn)
                new_vel_y = vel_y + delta_vel_y
                new_u = u + new_vel_y
            while (new_v < 0) or (new_v >= self.frame.shape[1] - 1):
                delta_vel_x = np.random.normal(0, self.sigma_dyn)
                new_vel_x = vel_x + delta_vel_x
                new_v = v + new_vel_x

            u = int(round(new_u))
            v = int(round(new_v))

            # update particle list to include new particle position
            self.particles[i] = [v, u, new_vel_x, new_vel_y, new_scale]

            # determine bounds for a comparison window from frame
            lower_u, upper_u = u - mid_y, u + mid_y
            lower_v, upper_v = v - mid_x, v + mid_x

            # check to see if we need to trim the window if particle close to
            #  edge of frame
            check_top, check_bot = mid_y, mid_y
            check_left, check_right = mid_x, mid_x

            if lower_u < 0:
                check_top = u - 0
                lower_u = 0
                resize_true = True
            if lower_v < 0:
                check_left = v - 0
                lower_v = 0
                resize_true = True
            if upper_u > frame.shape[0]:
                check_bot = frame.shape[0] - u
                upper_u = frame.shape[0]
                resize_true = True
            if upper_v > frame.shape[1]:
                check_right = frame.shape[1] - v
                upper_v = frame.shape[1]
                resize_true = True

            # resize template to match shrunken window in case we have
            # particles near edge of frame
            if resize_true:
                template_gray_comp = template_gray[mid_y - check_top:
                mid_y + check_bot, mid_x - check_left: mid_x + check_right]
                frame_cutout = frame[u - check_top:
                u + check_bot, v - check_left: v + check_right].astype(float)

                template_col_comp = template_col[mid_y - check_top:
                mid_y + check_bot, mid_x - check_left:
                                    mid_x + check_right].astype(float)
                frame_col_comp = frame_col[u - check_top:
                u + check_bot, v - check_left: v +
                                               check_right].astype(float)
            else:
                template_gray_comp = template_gray
                template_col_comp = template_col
                frame_cutout = frame[lower_u: upper_u, lower_v:
                upper_v].astype(float)
                frame_col_comp = frame_col[lower_u: upper_u, lower_v:
                upper_v].astype(float)

            # calculate error using template matching and color histogram
            # comparison as defined in MultiParticleFilter class
            w_i = self.get_error_metric_new(template_gray_comp, frame_cutout,
                                            template_col_comp, frame_col_comp)

            normalization_factor += w_i
            self.weights[i] = w_i

        # update tracking vals, weights, and resample; also check for occlusion
        self.update_tracking_vals(normalization_factor)

    def get_center_mass(self):
        """ Calculates the center of mass for the current set of particles. 

        Center of mass of particles is calculated by finding the mean of ((x,
        y, vx, vy, s) * weight) across all current particles. For the
        FollowParticleFilter class, this method now also calculates mean 
        velocities of particles after weight update is complete.

        Returns:
            v: x-coordinate of center of mass as a float
            u: y-coordinate of center of mass as a float
            vx: velocity in the x dimension as a float
            vy: velocity in the y dimension as a float
            scale: template rescaling proportion as a float
        """

        particles = self.get_particles()
        weights = self.get_weights()

        # initialize weighted means and average distance to these means
        x_weighted_mean = 0
        y_weighted_mean = 0
        avg_vel_x = 0
        avg_vel_y = 0
        avg_scale = 0

        # calc the weighted mean
        for i in range(self.num_particles):
            v, u, vel_x, vel_y, scale = particles[i]

            x_weighted_mean += v * weights.item(i)
            y_weighted_mean += u * weights.item(i)
            avg_vel_x += vel_x * weights.item(i)
            avg_vel_y += vel_y * weights.item(i)
            avg_scale += scale * weights.item(i)

        return x_weighted_mean, y_weighted_mean, avg_vel_x, avg_vel_y, \
               avg_scale

    def update_tracking_vals(self, normalization_factor):
        """ Update average displacement and average template error values.

        This method expands upon the approach used in MDParticleFilter to 
        accommodate the new state represented by MultiParticleFilter.

        The top 5% of un-normalized weights are averaged for the current
        frame, with a running window of these averages used to threshold
        whenever a template is found (along with a multiplicative factor
        self.w_threshold.

        If template does not meet threshold, self.estimate_next_frame is
        called to handle what is assumed to be occlusion. See docstring in
        that method for more information. If template does meet threshold,
        update weights, resample, and update tracking variables for the
        current best scale and the current location (x, y).

        """

        threshold = self.w_threshold

        avg_weight = np.mean(self.weights)

        # determine the top 5 percent of weights and use this to calculate a
        #  running average of top weights
        top_5_percent = np.ceil(0.95 * len(self.all_avg_weights)).astype(int)
        # if this if the first frame use the total average from the first frame
        if self.all_avg_weights == []:
            self.all_avg_weights.append(avg_weight)
        running_avg = np.mean(self.all_avg_weights[top_5_percent - 1:])

        # determine the current best similarity measure
        max_weight = np.max(self.weights)

        # if occlusion is detected after tracking the object for at least 10
        # frames
        if max_weight < running_avg * threshold and \
                        len(self.all_avg_weights) > 10:
            self.estimate_next_frame()
        else:
            # if occlusion is not detected then update normalize weights,
            # resample particles and update displacement tracking value
            self.all_avg_weights.append(avg_weight)

            # normalize weights
            self.weights = self.weights / normalization_factor

            # resample particles
            self.particles = self.resample_particles()

            # get best location and scale of object
            v, u, vx, vy, avg_scale = self.get_center_mass()

            # update all scales to include recent best rescaling
            self.all_scales.append(avg_scale)

            last_x, last_y = self.current_position

            self.current_position = (v, u)

            # update displacements
            diff_x, diff_y = self.current_position[0] - last_x, \
                             self.current_position[1] - last_y
            self.all_displacements.append((diff_x, diff_y))

            # update template to incorporate appearance changes
            self.template = self.update_template(self.template)

    def update_template(self, template_img):
        """This function updates the template using an Infinite Impulse Response
        filter by updating the template as a weighted sum of the current
        template and the frame cutout from the best curent location.

        This function is called in the base class process function, using an
        isinstance check to determine if object is an 'ApperanceModelPF' obj.

        Args:
            template_img (numpy.array): template image as grayscale float.

        Returns:
            updated_template (numpy.array): updated template as a weighted sum
                                            of recent best frame cutout and
                                            current template.
        """

        # initialize
        alpha = self.alpha
        beta = 1 - alpha

        # get a resized version of the template at the avg best scale from
        # previous frame to determine size of current frame
        best_template = cv2.resize(self.orig_template, dsize=None,
                                   fx=self.all_scales[-1],
                              fy=self.all_scales[-1],
                              interpolation=cv2.INTER_AREA)

        # convert frame cutout to grayscale using luma value (0.3, 0.58, 0.12)
        if best_template.shape[0] % 2 == 1:
            best_template = best_template[:-1, :]
        if best_template.shape[1] % 2 == 1:
            best_template = best_template[:, :-1]

        mid_x = best_template.shape[1] // 2
        mid_y = best_template.shape[0] // 2

        template = best_template.astype(float)

        # plot each particle as a white dot and calc the weighted mean (x, y)
        v, u = self.current_position
        u, v = int(round(u)), int(round(v))

        # determine bounds for a comparison window from frame
        lower_u, upper_u = u - mid_y, u + mid_y
        lower_v, upper_v = v - mid_x, v + mid_x

        # if frame bounds are possible then update template using scaled up
        # version of best frame cutout
        if lower_u > 0 and upper_u < self.frame.shape[0] \
                and lower_v > 0 and upper_v < self.frame.shape[1]:
            frame_cutout = self.frame[lower_u: upper_u,
                           lower_v: upper_v].astype(float)

            # scale up frame cutout to match original template size
            best_frame = cv2.resize(frame_cutout,
                                         dsize=(template_img.shape[1],
                                                template_img.shape[0]),
                                 interpolation=cv2.INTER_CUBIC).astype(
                np.uint8)
            output_template = cv2.addWeighted(best_frame, alpha,
                                              self.template.astype('uint8'),
                                              beta, 0)

        else:
            output_template = template_img

        return output_template

    def dist_error(self, test_x, test_y, new_x, new_y):
        """ Returns an error metric based simply on how distant particles are
        from a predicted location using simple euclidean distance.

        This metric is simply used to keep particles close to where the
        object is predicted to be, simulating "tracking" during
        occlusion.

        self.sigma_exp is passed in through experiment.py, and default value
        is set to 0.04, determined through experimentation.

        Args:
            test_x: x-coordinate of current particle as a float.
            test_y: y-coordinate of current particle as a float.
            new_x: x-coordinate of predicted location as a float.
            new_y: y-coordinate of predicted location as a float.
        Returns:
            error_measure: similarity value returned as a float.
        """

        # determine residual between template and frame cutout at each pixel
        diff_x, diff_y = test_x - new_x, test_y - new_y

        # calculate euclidean distance between locations
        dist = np.sqrt(diff_x ** 2 + diff_y ** 2)

        # convert dist to a similarity measure between particle and predicted
        #  loc
        error_measure = np.exp(-dist / (2 * self.sigma_exp ** 2))

        return error_measure

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method is overloaded to free the original render method from
        clutter using a field of isinstances.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        # initialize avg dist to predicted location
        dist = 0
        particles = self.get_particles()

        # calculate the predicted location of the object in the current frame
        # and also get proper rescaling for bounding box if MDParticleFilter
        v, u, vx, vy, avg_scale = self.get_center_mass()
        new_template = cv2.resize(self.template, dsize=None,
                                  fx=avg_scale, fy=avg_scale,
                                  interpolation=cv2.INTER_AREA)

        mid_x = new_template.shape[1] // 2
        mid_y = new_template.shape[0] // 2

        # define top left and bot-right points for window rectangle
        pt_1 = (int(round(v)) - mid_x,
                int(round(u)) - mid_y)
        pt_2 = (int(round(v)) + mid_x,
                int(round(u)) + mid_y)
        cv2.rectangle(frame_in, pt_1, pt_2, color=(255, 255, 0), thickness=2)

        # determine the average distance of all particles to mean (x, y) and
        # also draw the current position of each particle
        for i in range(self.num_particles):
            x, y, _, _, _ = particles[i]

            # draw the particle
            color = (0, 0, 255)  # set the color for each particle
            cv2.circle(frame_in, center=(int(round(x)), int(round(y))),
                       radius=2,
                       color=color,
                       thickness=1)

            x_dev = x - v
            y_dev = y - u

            dist += np.sqrt(x_dev ** 2 + y_dev ** 2) * self.weights.item(i)

        # draw circle around mean (x, y) with radius equal to avg dist
        cv2.circle(frame_in, center=(int(round(v)), int(round(u))),
                   radius=int(round(dist)),
                   color=(0, 0, 0),
                   thickness=2)

    def estimate_next_frame(self):
        """ Estimate the next frame if no reasonable template is found
        in the image, hence simulating tracking during occlusion.

        This method handles occlusion by updating new location of the object
        in the current frame to be a new predicted location, determined by
        extrapolating from previously tracked average displacements.

        The ideal scale for the template also continues to scale down each
        frame by 0.99 to approximate the size of the object getting smaller
        by a small amount after occlusion.

        Particles are still resampled during occlusion, and this is done so
        as to allow for a varied distribution around the predicted point to
        hopefully pick up the object after occlusion with the right (x, y, s).

        """

        # initialize
        normalization_factor = 0.0

        # update new location if template has been previously successfully
        # tracked
        try:
            avg_displace_x, avg_displace_y = np.mean(self.all_displacements[
                                                  -10:], axis=0)

            new_x = self.current_position[0] + avg_displace_x
            new_y = self.current_position[1] + avg_displace_y
        except:
            avg_displace_x, avg_displace_y = 0, 0
            new_x, new_y = self.current_position

        # use previous scale to determine next scale
        new_scale = self.all_scales[-1] * 0.99

        # add this new scale to the tracking list of rescale values
        self.all_scales.append(new_scale)

        self.current_position = (new_x, new_y)

        # re-distribute particles about the predicted new location using the
        # same gaussian as dynamics model
        means = [int(round(new_x)), int(round(new_y)), avg_displace_x,
                 avg_displace_y]
        cov = [[self.sigma_dyn, 0, 0, 0], [0, self.sigma_dyn, 0, 0],
               [0, 0, self.sigma_dyn, 0], [0, 0, 0, self.sigma_dyn]]
        scales = np.random.randn(self.num_particles, 1) * self.sigma_scale + \
                 new_scale

        self.particles = np.random.multivariate_normal(means, cov,
                                                       self.num_particles).astype(
            int)
        self.particles = np.hstack((self.particles, scales))

        # update weights based on distance to predicted location
        for i in range(self.num_particles):
            test_x, test_y, _, _, _ = self.particles[i]

            w_i = self.dist_error(test_x, test_y, new_x, new_y)

            normalization_factor += w_i

            self.weights[i] = w_i

        # normalize weights
        self.weights = self.weights / normalization_factor

        # resample particles
        self.particles = self.resample_particles()