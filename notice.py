#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Notices during runtime.

:copyright: 2015 Agile Geoscience
:license: Apache 2.0
"""


class Notice(object):
    """
    Helper class to make printout more readable.
    """
    styles = {'HEADER': '\033[95m',
              'INFO': '\033[94m',      # blue
              'OK': '\033[92m',        # green
              'WARNING': '\033[93m',   # red
              'FAIL': '\033[91m',
              'BOLD': '\033[1m'
              }
    ENDC = '\033[0m'

    def __init__(self, string, style, hold=False):
        string = self.styles[style.upper()] + string + self.ENDC
        end = '' if hold else '\n'
        print(string, end=end)

    @classmethod
    def title(cls):
        """Makes a logo."""
        logo = """
Welcome to
┌─┐┌─┐┬┌─┐┌─┐┬  ┌─┐┌┬┐
└─┐├┤ │└─┐├─┘│  │ │ │ 
└─┘└─┘┴└─┘┴  ┴─┘└─┘ ┴ 
             Good luck"""
        return cls(logo, 'WARNING')

    @classmethod
    def warning(cls, string, hold=False):
        """Yellow."""
        return cls(string, 'WARNING', hold=hold)

    @classmethod
    def fail(cls, string, hold=False):
        """Red."""
        return cls(string, 'FAIL', hold=hold)

    @classmethod
    def header(cls, string, hold=False):
        """Pink."""
        return cls('\n'+string+'\n', 'HEADER', hold=hold)

    @classmethod
    def hr_header(cls, string, hold=False):
        """Pink."""
        hr = "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
        return cls(hr+string.upper(), 'HEADER', hold=hold)

    @classmethod
    def info(cls, string, hold=False):
        """Blue."""
        return cls(string, 'INFO', hold=hold)

    @classmethod
    def ok(cls, string, hold=False):
        """Green."""
        return cls(string, 'OK', hold=hold)
