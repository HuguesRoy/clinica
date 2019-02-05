# coding: utf8

# WARNING: Don't put any import statement here except if it's absolutely
# necessary. Put it *inside* the different methods.
# Otherwise it will slow down the dynamic loading of the pipelines list by the
# command line tool.
import clinica.pipelines.engine as cpe

__author__ = "Simona Bottani"
__copyright__ = "Copyright 2016-2019 The Aramis Lab Team"
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Simona Bottani"
__email__ = "simona.bottani@icm-institute.org"
__status__ = "Development"


class SpatialSVM(cpe.Pipeline):
    """SpatialSVM - Prepare input data for SVM with spatial and anatomical regularization.

    Todos:
        - [ ] Final version of CAPS.
        - [ ] Remove --voxel_size flag and detect automatically this parameter.

    Args:
        input_dir: A BIDS directory.
        output_dir: An empty output directory where CAPS structured data will be written.
        subjects_sessions_list: The Subjects-Sessions list file (in .tsv format).

    Returns:
        A clinica pipeline object containing the SpatialSVM pipeline.

    Raises:
    """

    def check_custom_dependencies(self):
        """Check dependencies that can not be listed in the `info.json` file.
        """
        pass

    def get_input_fields(self):
        """Specify the list of possible inputs of this pipeline.

        Returns:
            A list of (string) input fields name.
        """

        return ['dartel_input', 'input_image']

    def get_output_fields(self):
        """Specify the list of possible outputs of this pipeline.

        Returns:
            A list of (string) output fields name.
        """

        return ['regularized_image']

    def build_input_node(self):
        """Build and connect an input node to the pipeline.
        """

        import nipype.interfaces.utility as nutil
        import nipype.pipeline.engine as npe
        from clinica.utils.stream import cprint
        from os.path import join, exists

        # This node is supposedly used to load BIDS inputs when this pipeline is
        # not already connected to the output of a previous Clinica pipeline.
        # For the purpose of the example, we simply read input arguments given
        # by the command line interface and transmitted here through the
        # `self.parameters` dictionary and pass it to the `self.input_node` to
        # further by used as input of the
        # core nodes.

        read_parameters_node = npe.Node(name="LoadingCLIArguments",
                                        interface=nutil.IdentityInterface(fields=self.get_input_fields(),
                                                                          mandatory_inputs=True))
        image_type = self.parameters['image_type']
        pet_type = self.parameters['pet_type']

        participant_id_pycaps = [sub[4:] for sub in self.subjects]
        session_id_pycaps = [ses[4:] for ses in self.sessions]

        input_image = []
        if image_type == 't1':
            subjects_not_found = []
            for i, sub in enumerate(self.subjects):
                input_image_single_subject = join(self.caps_directory,
                                                  'subjects', sub, self.sessions[i], 't1/spm/dartel/group-'
                                                  + self.parameters['group_id'], sub + '_' + self.sessions[i]
                                                  + '_T1w_segm-graymatter_space-Ixi549Space_modulated-on_probability.nii.gz')
                if not exists(input_image_single_subject):
                    subjects_not_found.append(input_image_single_subject)
                else:
                    input_image.append(input_image_single_subject)

            if len(subjects_not_found) > 0:
                error_string = ''
                for file in subjects_not_found:
                    error_string = error_string + file + '\n'
                raise IOError('Following files were not found :\n' + error_string)


        elif image_type == 'pet':
            input_image = caps_layout.get(return_type='file',
                                          subject=subjects_regex,
                                          session=sessions_regex,
                                          group_id=self.parameters['group_id'],
                                          pet_file=pet_type
                                          )
        else:
            raise IOError('Image type ' + image_type + ' unknown')

        if len(input_image) != len(self.subjects):
            raise IOError(str(len(input_image)) + ' file(s) found. ' + str(len(self.subjects)) + ' are expected')

        dartel_input = [join(self.caps_directory,
                             'groups',
                             'group-' + self.parameters['group_id'],
                             't1',
                             'group-' + self.parameters['group_id'] + '_template.nii.gz')]
        if not exists(dartel_input[0]):
            raise IOError('Dartel Input ' + dartel_input[0] + ' does not seem to exist')

        read_parameters_node.inputs.dartel_input = dartel_input
        read_parameters_node.inputs.input_image = input_image

        self.connect([
            (read_parameters_node,      self.input_node,    [('dartel_input',    'dartel_input')]),
            (read_parameters_node,      self.input_node,    [('input_image',    'input_image')])

        ])

    def build_output_node(self):
        """Build and connect an output node to the pipeline.
        """

        # In the same idea as the input node, this output node is supposedly
        # used to write the output fields in a CAPS. It should be executed only
        # if this pipeline output is not already connected to a next Clinica
        # pipeline.

        pass

    def build_core_nodes(self):
        """Build and connect the core nodes of the pipeline.
        """

        import clinica.pipelines.machine_learning_spatial_svm.spatial_svm_utils as utils
        import nipype.interfaces.utility as nutil
        import nipype.pipeline.engine as npe
        import nipype.interfaces.io as nio

        fisher_tensor_generation = npe.Node(name="obtain_g_fisher_tensor",
                                            interface=nutil.Function(input_names=['dartel_input', 'FWHM'],
                                                                     output_names=['fisher_tensor', 'fisher_tensor_path'],
                                                                     function=utils.obtain_g_fisher_tensor))
        fisher_tensor_generation.inputs.FWHM = self.parameters['fwhm']

        time_step_generation = npe.Node(name='estimation_time_step',
                                        interface=nutil.Function(input_names=['dartel_input', 'FWHM', 'g'],
                                                                 output_names=['t_step', 'json_file'],
                                                                 function=utils.obtain_time_step_estimation))
        time_step_generation.inputs.FWHM = self.parameters['fwhm']

        heat_solver_equation = npe.MapNode(name='heat_solver_equation',
                                           interface=nutil.Function(input_names=['input_image', 'g',
                                                                                 'FWHM', 't_step', 'dartel_input'],
                                                                    output_names=['regularized_image'],
                                                                    function=utils.heat_solver_equation),
                                           iterfield=['input_image'])
        heat_solver_equation.inputs.FWHM = self.parameters['fwhm']

        datasink = npe.Node(nio.DataSink(),
                            name='sinker')
        datasink.inputs.base_directory = self.caps_directory
        datasink.inputs.parameterization = True
        if self.parameters['image_type'] == 't1':
            datasink.inputs.regexp_substitutions = [
                (r'(.*)/regularized_image/.*/(.*(sub-(.*)_ses-(.*))_T1w(.*)_(probability.*))$',
                 r'\1/subjects/sub-\4/ses-\5/t1/input_regularised_svm/group-' + self.parameters['group_id'] + r'/\3\6_regularization-Fisher_fwhm-' + str(self.parameters['fwhm']) + r'_\7'),
                (r'(.*)json_file/(output_data.json)$',
                    r'\1/groups/group-' + self.parameters['group_id'] + r'/t1/input_regularised_svm/group-' + self.parameters['group_id'] + r'_space-Ixi549Space_modulated-on_regularization-Fisher_fwhm-'
                    + str(self.parameters['fwhm']) + r'_parameters.json'),

                (r'(.*)fisher_tensor_path/(output_fisher_tensor.npy)$',
                    r'\1/groups/group-' + self.parameters['group_id'] + r'/t1/input_regularised_svm/group-' + self.parameters[
                        'group_id'] + r'_space-Ixi549Space_modulated-on_regularization-Fisher_fwhm-'
                    + str(self.parameters['fwhm']) + r'_gram.npy')
            ]

        elif self.parameters['image_type'] == 'pet':
            datasink.inputs.regexp_substitutions = [
                (r'(.*)/regularized_image/.*/(.*(sub-(.*)_ses-(.*))_(task.*)_(pet.*))$',
                 r'\1/subjects/sub-\4/ses-\5/pet/input_regularised_svm/group-' + self.parameters[
                     'group_id'] + r'/\3\6_regularization-Fisher_fwhm-' + str(self.parameters['fwhm']) + r'_\7'),

                (r'(.*)json_file/(output_data.json)$',
                 r'\1/groups/group-' + self.parameters['group_id'] + r'/t1/input_regularised_svm/group-' +
                 self.parameters['group_id'] + r'_space-Ixi549Space_modulated-on_regularization-Fisher_fwhm-'
                 + str(self.parameters['fwhm']) + r'_parameters.json'),
                (r'(.*)fisher_tensor_path/(output_fisher_tensor.npy)$',
                 r'\1/groups/group-' + self.parameters['group_id'] + r'/t1/input_regularised_svm/group-' +
                 self.parameters[
                     'group_id'] + r'_space-Ixi549Space_modulated-on_regularization-Fisher_fwhm-'
                 + str(self.parameters['fwhm']) + r'_gram.npy')
            ]

        # Connection
        # ==========
        self.connect([
            (self.input_node,      fisher_tensor_generation,    [('dartel_input',    'dartel_input')]),
            (fisher_tensor_generation,      time_step_generation,    [('fisher_tensor',    'g')]),

            (self.input_node, time_step_generation, [('dartel_input', 'dartel_input')]),
            (self.input_node, heat_solver_equation, [('input_image', 'input_image')]),
            (fisher_tensor_generation, heat_solver_equation, [('fisher_tensor', 'g')]),
            (time_step_generation, heat_solver_equation, [('t_step', 't_step')]),
            (self.input_node, heat_solver_equation, [('dartel_input', 'dartel_input')]),

            (fisher_tensor_generation, datasink, [('fisher_tensor_path', 'fisher_tensor_path')]),
            (time_step_generation, datasink, [('json_file', 'json_file')]),
            (heat_solver_equation, datasink, [('regularized_image', 'regularized_image')])
        ])
