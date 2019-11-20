# coding: utf8

import clinica.engine as ce


class T1VolumeNewTemplateCLI(ce.CmdParser):

    def define_name(self):
        """Define the sub-command name to run this pipeline."""
        self._name = 't1-volume'

    def define_description(self):
        """Define a description of this pipeline."""
        self._description = ('Volume-based processing of T1-weighted MR images:\n'
                             'http://clinica.run/doc/Pipelines/T1_Volume/')

    def define_options(self):
        """Define the sub-command arguments."""
        from clinica.engine.cmdparser import PIPELINE_CATEGORIES
        # Clinica compulsory arguments (e.g. BIDS, CAPS, group_id)
        clinica_comp = self._args.add_argument_group(PIPELINE_CATEGORIES['CLINICA_COMPULSORY'])
        clinica_comp.add_argument("bids_directory",
                                  help='Path to the BIDS directory.')
        clinica_comp.add_argument("caps_directory",
                                  help='Path to the CAPS directory.')
        clinica_comp.add_argument("group_id",
                                  help='User-defined identifier for the provided group of subjects.')
        # Optional arguments (e.g. FWHM)
        optional = self._args.add_argument_group(PIPELINE_CATEGORIES['OPTIONAL'])
        optional.add_argument("-s", "--smooth",
                              nargs='+', type=int, default=[8],
                              help="A list of integers specifying the different isomorphic FWHM in millimeters "
                                   "to smooth the image (default: --smooth 8).")
        # Clinica standard arguments (e.g. --n_procs)
        self.add_clinica_standard_arguments()
        # Advanced arguments (i.e. tricky parameters)
        advanced = self._args.add_argument_group(PIPELINE_CATEGORIES['ADVANCED'])
        advanced.add_argument("-tc", "--tissue_classes",
                              metavar='', nargs='+', type=int, default=[1, 2, 3], choices=range(1, 7),
                              help="Tissue classes (1: gray matter (GM), 2: white matter (WM), "
                                   "3: cerebrospinal fluid (CSF), 4: bone, 5: soft-tissue, 6: background) to save "
                                   "(default: GM, WM and CSF i.e. --tissue_classes 1 2 3).")
        advanced.add_argument("-dt", "--dartel_tissues",
                              metavar='', nargs='+', type=int, default=[1, 2, 3], choices=range(1, 7),
                              help='Tissues to use for DARTEL template calculation '
                                   '(default: GM, WM and CSF i.e. --dartel_tissues 1 2 3).')
        advanced.add_argument("-tpm", "--tissue_probability_maps",
                              metavar='TissueProbabilityMap.nii',
                              help='Tissue probability maps to use for segmentation (default: TPM from SPM software).')
        advanced.add_argument("-swu", "--save_warped_unmodulated",
                              action='store_true', default=True,
                              help="Save warped unmodulated images for tissues specified in --tissue_classes flag.")
        advanced.add_argument("-swm", "--save_warped_modulated",
                              action='store_true',
                              help="Save warped modulated images for tissues specified in --tissue_classes flag.")
        advanced.add_argument("-m", "--modulate",
                              type=bool, default=True,
                              metavar=('True/False'),
                              help='A boolean. Modulate output images - no modulation preserves concentrations '
                                   '(default: --modulate True).')
        advanced.add_argument("-vs", "--voxel_size",
                              metavar=('float'),
                              nargs=3, type=float,
                              help="A list of 3 floats specifying the voxel sizeof the output image "
                                   "(default: --voxel_size 1.5 1.5 1.5).")
        list_atlases = ['AAL2', 'LPBA40', 'Neuromorphometrics', 'AICHA', 'Hammers']
        advanced.add_argument("-atlases", "--atlases",
                              nargs='+', type=str, metavar='',
                              default=list_atlases, choices=list_atlases,
                              help='A list of atlases used to calculate the regional mean GM concentrations (default: '
                                   'all atlases i.e. --atlases AAL2 AICHA Hammers LPBA40 Neuromorphometrics).')

    def run_command(self, args):
        """Run the pipeline with defined args."""
        from colorama import Fore
        from networkx import Graph
        from ..t1_volume_tissue_segmentation.t1_volume_tissue_segmentation_pipeline import T1VolumeTissueSegmentation
        from ..t1_volume_create_dartel.t1_volume_create_dartel_pipeline import T1VolumeCreateDartel
        from ..t1_volume_dartel2mni.t1_volume_dartel2mni_pipeline import T1VolumeDartel2MNI
        from ..t1_volume_parcellation.t1_volume_parcellation_pipeline import T1VolumeParcellation
        from clinica.utils.check_dependency import verify_cat12_atlases
        from clinica.utils.stream import cprint
        from clinica.utils.ux import print_end_pipeline, print_crash_files_and_exit

        # Initialization
        # ==============
        # If the user wants to use any of the atlases of CAT12 and has not installed it, we just remove it from the list
        # of the computed atlases
        args.atlases = verify_cat12_atlases(args.atlases)

        parameters = {
            'tissue_classes': args.tissue_classes,
            'dartel_tissues': args.dartel_tissues,
            'tissue_probability_maps': args.tissue_probability_maps,
            'save_warped_unmodulated': args.save_warped_unmodulated,
            'save_warped_modulated': args.save_warped_modulated,
            'voxel_size': tuple(args.voxel_size) if args.voxel_size is not None else None,
            'modulation': args.modulate,
            'fwhm': args.smooth,
            'atlas_list': args.atlases
        }

        cprint(
            'The t1-volume pipeline is divided into 4 parts:'
            '\t%st1-volume-tissue-segmentation pipeline%s: Tissue segmentation, bias correction and spatial normalization to MNI space'
            '\t%st1-volume-create-dartel pipeline%s: Inter-subject registration with the creation of a new DARTEL template'
            '\t%st1-volume-dartel2mni pipeline%s: DARTEL template to MNI'
            '\t%st1-volume-parcellation pipeline%s: Atlas statistics'
            % (Fore.BLUE, Fore.RESET, Fore.BLUE, Fore.RESET, Fore.BLUE, Fore.RESET, Fore.BLUE, Fore.RESET)
        )

        # t1-volume-segmentation
        # ======================
        cprint('%sPart 1/4: Running t1-volume-segmentation pipeline%s' % (Fore.BLUE, Fore.RESET))
        tissue_segmentation_pipeline = T1VolumeTissueSegmentation(
            bids_directory=self.absolute_path(args.bids_directory),
            caps_directory=self.absolute_path(args.caps_directory),
            tsv_file=self.absolute_path(args.subjects_sessions_tsv),
            base_dir=self.absolute_path(args.working_directory),
            parameters=parameters,
            name="t1-volume-tissue-segmentation"
        )

        if args.n_procs:
            exec_pipeline = tissue_segmentation_pipeline.run(plugin='MultiProc',
                                                             plugin_args={'n_procs': args.n_procs})
        else:
            exec_pipeline = tissue_segmentation_pipeline.run()

        if isinstance(exec_pipeline, Graph):
            print_end_pipeline(self.name,
                               tissue_segmentation_pipeline.base_dir,
                               tissue_segmentation_pipeline.base_dir_was_specified)
        else:
            print_crash_files_and_exit(args.logname,
                                       tissue_segmentation_pipeline.base_dir)

        # t1-volume-create-dartel
        # =======================
        cprint('%sPart 2/4: Running t1-volume-create-dartel pipeline%s' % (Fore.BLUE, Fore.RESET))
        create_dartel_pipeline = T1VolumeCreateDartel(
            bids_directory=self.absolute_path(args.bids_directory),
            caps_directory=self.absolute_path(args.caps_directory),
            tsv_file=self.absolute_path(args.subjects_sessions_tsv),
            base_dir=self.absolute_path(args.working_directory),
            parameters=parameters,
            name="t1-volume-create-dartel"
        )

        if args.n_procs:
            exec_pipeline = create_dartel_pipeline.run(plugin='MultiProc',
                                                       plugin_args={'n_procs': args.n_procs})
        else:
            exec_pipeline = create_dartel_pipeline.run()

        if isinstance(exec_pipeline, Graph):
            print_end_pipeline(self.name,
                               create_dartel_pipeline.base_dir,
                               create_dartel_pipeline.base_dir_was_specified)
        else:
            print_crash_files_and_exit(args.logname,
                                       create_dartel_pipeline.base_dir)

        # t1-volume-dartel2mni
        # ====================
        cprint('%sPart 3/4: Running t1-volume-dartel2mni pipeline%s' % (Fore.BLUE, Fore.RESET))
        dartel2mni_pipeline = T1VolumeDartel2MNI(
            bids_directory=self.absolute_path(args.bids_directory),
            caps_directory=self.absolute_path(args.caps_directory),
            tsv_file=self.absolute_path(args.subjects_sessions_tsv),
            base_dir=self.absolute_path(args.working_directory),
            parameters=parameters,
            name="t1-volume-dartel2mni"
        )

        if args.n_procs:
            exec_pipeline = dartel2mni_pipeline.run(plugin='MultiProc',
                                                    plugin_args={'n_procs': args.n_procs})
        else:
            exec_pipeline = dartel2mni_pipeline.run()

        if isinstance(exec_pipeline, Graph):
            print_end_pipeline(self.name,
                               dartel2mni_pipeline.base_dir,
                               dartel2mni_pipeline.base_dir_was_specified)
        else:
            print_crash_files_and_exit(args.logname,
                                       dartel2mni_pipeline.base_dir)

        # t1-volume-parcellation
        # ======================
        cprint('%sPart 4/4: Running t1-volume-parcellation pipeline%s' % (Fore.BLUE, Fore.RESET))
        parcellation_pipeline = T1VolumeParcellation(
            bids_directory=self.absolute_path(args.bids_directory),
            caps_directory=self.absolute_path(args.caps_directory),
            tsv_file=self.absolute_path(args.subjects_sessions_tsv),
            base_dir=self.absolute_path(args.working_directory),
            parameters=parameters,
            name="t1-volume-parcellation"
        )

        if args.n_procs:
            exec_pipeline = parcellation_pipeline.run(plugin='MultiProc',
                                                      plugin_args={'n_procs': args.n_procs})
        else:
            exec_pipeline = parcellation_pipeline.run()

        if isinstance(exec_pipeline, Graph):
            print_end_pipeline(self.name,
                               parcellation_pipeline.base_dir,
                               parcellation_pipeline.base_dir_was_specified)
        else:
            print_crash_files_and_exit(args.logname,
                                       parcellation_pipeline.base_dir)