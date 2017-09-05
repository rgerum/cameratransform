"""
Use the codeship environmental variables to fill in the right values to upload
to coveralls.
"""

import os
from coveralls_hg.api import API

# pylint:disable=dangerous-default-value
def main(env=os.environ, coverage_file='.coverage'):
    "main script"
    user, repo = env['CI_REPO_NAME'].split('/')
    api = API(user,repo, token=env['COVERALLS_REPO_TOKEN'])

    api.set_build_values(build_url=env['BITBUCKET_CLONE_DIR'], branch=env['BITBUCKET_BRANCH'])

    api.set_dvcs_commit(commit_id=env['BITBUCKET_COMMIT'],
                        message="", branch=env['BITBUCKET_BRANCH'])

    api.set_dvcs_user(name_author="",
                      email_author="")

    api.set_service_values(number=env['BITBUCKET_BUILD_NUMBER'])

    cwd = os.path.abspath(os.getcwd())
    api.set_source_files(coverage_file, strip_path=cwd)

    api.upload_coverage()


if __name__ == '__main__': # pragma: no cover
    main()

