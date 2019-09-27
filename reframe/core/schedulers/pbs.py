#
# PBS backend
#
# - Initial version submitted by Rafael Escovar, ASML
#

import os
import itertools
import re
import time
from datetime import datetime
from random import shuffle

import reframe.core.schedulers as sched
import reframe.utility.os_ext as os_ext
from reframe.core.config import settings
from reframe.core.exceptions import SpawnedProcessError, JobError
from reframe.core.logging import getlogger
from reframe.core.schedulers.registry import register_scheduler


# Time to wait after a job is finished for its standard output/error to be
# written to the corresponding files.
PBS_OUTPUT_WRITEBACK_WAIT = 3


@register_scheduler('pbs')
class PbsJob(sched.Job):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefix = '#PBS'
        self._time_finished = None

        # Optional part of the job id refering to the PBS server
        self._pbs_server = None

    def _emit_lselect_option(self):
        num_tasks_per_node = self._num_tasks_per_node or 1
        num_cpus_per_task = self._num_cpus_per_task or 1
        num_nodes = self._num_tasks // num_tasks_per_node
        num_cpus_per_node = num_tasks_per_node * num_cpus_per_task
        if self.sched_nodelist:
            nodelist = self.sched_nodelist.split('+')
            if len(nodelist) < num_nodes:
                raise JobError('Insufficient number of nodes '
                               'in the nodelist')
            shuffle(nodelist)
            select_opt = '-l nodes=%s:ppn=%s' % (nodelist[0], num_cpus_per_node)
            for x in range(1,num_nodes):
                select_opt = select_opt + '+%s:ppn=%s' % (nodelist[x], num_tasks_per_node)
        else:
            select_opt = '-l nodes=%s:ppn=%s' % (num_nodes, num_cpus_per_node)

        # Options starting with `-` are emitted in separate lines
        rem_opts = []
        verb_opts = []
        for opt in (*self.sched_access, *self.options):
            if opt.startswith('-'):
                rem_opts.append(opt)
            elif opt.startswith('#'):
                verb_opts.append(opt)
            else:
                select_opt += ':' + opt

        return [self._format_option(select_opt),
                *(self._format_option(opt) for opt in rem_opts),
                *verb_opts]

    def _format_option(self, option):
        return self._prefix + ' ' + option

    def _run_command(self, cmd, timeout=None):
        """Run command cmd and re-raise any exception as a JobError."""
        try:
            return os_ext.run_command(cmd, check=True, timeout=timeout)
        except SpawnedProcessError as e:
            raise JobError(jobid=self._jobid) from e

    def emit_preamble(self):
        preamble = [
            self._format_option('-N "%s"' % self.name),
            self._format_option('-o %s' % self.stdout),
            self._format_option('-e %s' % self.stderr),
        ]

        if self.time_limit is not None:
            preamble.append(
                self._format_option('-l walltime=%d:%d:%d' % self.time_limit))

        if self.sched_partition:
            preamble.append(
                self._format_option('-q %s' % self.sched_partition))

        if self.sched_account:
            preamble.append(
                self._format_option('-A %s' % self.sched_account))

        preamble += self._emit_lselect_option()

        # PBS starts the job in the home directory by default
        preamble.append('\ncd $PBS_O_WORKDIR')
        return preamble

    def get_all_nodes(self):
        raise NotImplementedError('pbs backend does not support node listing')

    def filter_nodes(self, nodes, options):
        raise NotImplementedError('pbs backend does not support '
                                  'node filtering')

    def submit(self):
        # `-o` and `-e` options are only recognized in command line by the PBS
        # Slurm wrappers.
        cmd = 'qsub -o %s -e %s %s' % (self.stdout, self.stderr,
                                       self.script_filename)
        completed = self._run_command(cmd, settings().job_submit_timeout)
        jobid_match = re.search(r'^(?P<jobid>\S+)', completed.stdout)
        if not jobid_match:
            raise JobError('could not retrieve the job id '
                           'of the submitted job')

        jobid, *info = jobid_match.group('jobid').split('.', maxsplit=2)
        self._jobid = int(jobid)
        if info:
            self._pbs_server = info[0]

    def wait(self):
        super().wait()
        intervals = itertools.cycle(settings().job_poll_intervals)
        while not self.finished():
            time.sleep(next(intervals))

    def cancel(self):
        super().cancel()

        # Recreate the full job id
        jobid = str(self._jobid)
        if self._pbs_server:
            jobid += '.' + self._pbs_server

        getlogger().debug('cancelling job (id=%s)' % jobid)
        self._run_command('qdel %s' % jobid, settings().job_submit_timeout)

    def _get_nodelist(self, jobstat):
        # exec_host = o0580/0-27+o0444/0-27+o0345/0-27
        nodelist_match = re.search('exec_host = (?P<nodelist>\S+)', jobstat.stdout)
        if nodelist_match:
            nodelist = [ x.split('/')[0] for x in nodelist_match.group('nodelist').split('+') ]
            nodelist.sort()
            self._nodelist = nodelist

    def finished(self):
        super().finished()
        cmd = 'qstat -f %s' % (str(self._jobid))
        completed = self._run_command(cmd)
        jobstat_match = re.search('job_state = (?P<jobstat>\S+)', completed.stdout)
        job_done = False
        if not jobstat_match:
            job_done = True
        else:
            jobstat, *info = jobstat_match.group('jobstat').split('.', maxsplit=2)
            if jobstat != 'R' and jobstat != 'Q':
                job_done = True
        with os_ext.change_dir(self.workdir):
            done = os.path.exists(self.stdout) and os.path.exists(self.stderr) and job_done

        if done:
            t_now = datetime.now()
            self._time_finished = self._time_finished or t_now
            time_from_finish = (t_now - self._time_finished).total_seconds()
            self._get_nodelist(completed)

        return done and time_from_finish > PBS_OUTPUT_WRITEBACK_WAIT
