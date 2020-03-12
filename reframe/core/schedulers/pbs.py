#
# PBS backend
#
# - Initial version submitted by Rafael Escovar, ASML
#

import functools
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

JOB_STATES = {
    'Q': 'QUEUED',
    'H': 'HELD',
    'R': 'RUNNING',
    'E': 'EXITING',
    'T': 'MOVED',
    'W': 'WAITING',
    'S': 'SUSPENDED',
    'C': 'COMPLETED',
}

_run_strict = functools.partial(os_ext.run_command, check=True)


@register_scheduler('pbs')
class PbsJobScheduler(sched.JobScheduler):
    def __init__(self):
        self._prefix = '#PBS'
        self._time_finished = None

        # Optional part of the job id refering to the PBS server
        self._pbs_server = None

    def completion_time(self, job):
        return None

    def _emit_lselect_option(self, job):
        num_tasks_per_node = job.num_tasks_per_node or 1
        num_cpus_per_task = job.num_cpus_per_task or 1
        num_nodes = job.num_tasks // num_tasks_per_node
        num_cpus_per_node = num_tasks_per_node * num_cpus_per_task
        if job.sched_nodelist:
            nodelist = job.sched_nodelist.split('+')
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
        for opt in (*job.sched_access, *job.options):
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

    def emit_preamble(self, job):
        preamble = [
            self._format_option('-N "%s"' % job.name),
            self._format_option('-o %s' % job.stdout),
            self._format_option('-e %s' % job.stderr),
        ]

        if job.time_limit is not None:
            preamble.append(
                self._format_option('-l walltime=%d:%d:%d' % job.time_limit))

        if job.sched_partition:
            preamble.append(
                self._format_option('-q %s' % job.sched_partition))

        if job.sched_account:
            preamble.append(
                job._format_option('-A %s' % job.sched_account))

        preamble += self._emit_lselect_option(job)

        # PBS starts the job in the home directory by default
        preamble.append('\ncd $PBS_O_WORKDIR')
        #preamble.append('cd %s' % job.workdir)
        return preamble

    def allnodes(self):
        raise NotImplementedError('pbs backend does not support node listing')

    def filternodes(self, job, nodes):
        raise NotImplementedError('pbs backend does not support '
                                  'node filtering')

    def submit(self, job):
        # `-o` and `-e` options are only recognized in command line by the PBS
        # Slurm wrappers.
        cmd = 'qsub -o %s -e %s %s' % (job.stdout, job.stderr,
                                       job.script_filename)
        completed = _run_strict(cmd, timeout=settings().job_submit_timeout)
        jobid_match = re.search(r'^(?P<jobid>\S+)', completed.stdout)
        if not jobid_match:
            raise JobError('could not retrieve the job id '
                           'of the submitted job')

        jobid, *info = jobid_match.group('jobid').split('.', maxsplit=1)
        job.jobid = int(jobid)
        if info:
            self._pbs_server = info[0]

    def wait(self, job):
        intervals = itertools.cycle(settings().job_poll_intervals)
        while not self.finished(job):
            time.sleep(next(intervals))

    def cancel(self, job):
        # Recreate the full job id
        jobid = str(job.jobid)
        if self._pbs_server:
            jobid += '.' + self._pbs_server

        getlogger().debug('cancelling job (id=%s)' % jobid)
        _run_strict('qdel %s' % jobid, timeout=settings().job_submit_timeout)

    def _get_nodelist(self, job, stdout):
        # exec_host = o0580/0-27+o0444/0-27+o0345/0-27
        nodelist_match = re.search('exec_host = (?P<nodelist>\S+)', stdout)
        if nodelist_match:
            nodelist = [ x.split('/')[0] for x in nodelist_match.group('nodelist').split('+') ]
            nodelist.sort()
            job.nodelist = nodelist

    def _update_state(self, job):
        '''Check the status of the job.'''

        completed = os_ext.run_command('qstat -f %s' % job.jobid)

        # Depending on the configuration, completed jobs will remain on the job
        # list for a limited time, or be removed upon completion.
        # If qstat cannot find the jobid, it returns code 153.
        if completed.returncode == 153:
            getlogger().debug(
                'jobid not known by scheduler, assuming job completed'
            )
            job.state = 'COMPLETED'
            return                  # COMPLETED

        if completed.returncode != 0:
            raise JobError('qstat failed: %s' % completed.stderr, job.jobid)

        # Update nodelist
        self._get_nodelist(job, completed.stdout)

        state_match = re.search(
            r'^\s*job_state = (?P<state>[A-Z])', completed.stdout, re.MULTILINE
        )
        if not state_match:
            getlogger().debug(
                'job state not found (stdout follows)\n%s' % completed.stdout
            )
            return

        state = state_match.group('state')
        job.state = JOB_STATES[state]
        if job.state == 'COMPLETED':
            code_match = re.search(
                r'^\s*exit_status = (?P<code>\d+)',
                completed.stdout,
                re.MULTILINE,
            )
            if not code_match:
                return              # COMPLETED

            job.exitcode = int(code_match.group('code'))

    def finished(self, job):
        try:
            self._update_state(job)
        except JobError as e:
            # We ignore these exceptions at this point and we simply mark the
            # job as unfinished.
            getlogger().debug('ignoring error during polling: %s' % e)
            return False
        else:
            return job.state == 'COMPLETED'
"""
    def finished(self, job):
        cmd = 'qstat -f %s' % (str(job.jobid))
        completed = _run_strict(cmd, timeout=settings().job_submit_timeout)
        jobstat_match = re.search('job_state = (?P<jobstat>\S+)', completed.stdout)
        job_done = False
        if not jobstat_match:
            job_done = True
        else:
            jobstat, *info = jobstat_match.group('jobstat').split('.', maxsplit=1)
            if jobstat != 'R' and jobstat != 'Q':
                job_done = True
        with os_ext.change_dir(job.workdir):
            done = os.path.exists(job.stdout) and os.path.exists(job.stderr) and job_done

        if done:
            t_now = datetime.now()
            self._time_finished = self._time_finished or t_now
            time_from_finish = (t_now - self._time_finished).total_seconds()
            self._get_nodelist(job, completed)

        return done and time_from_finish > PBS_OUTPUT_WRITEBACK_WAIT
"""
